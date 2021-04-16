import sys
sys.path.insert(0,'./')

from argparse import ArgumentParser
import pytorch_lightning as pl

import json
from typing import Iterator, List, Dict, Optional
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# for dataset reader
from allennlp.data.data_loaders import MultiProcessDataLoader, SimpleDataLoader
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.vocabulary import Vocabulary

# read pretrained embedding from AWS S3
from allennlp.modules.token_embedders.embedding import _read_embeddings_from_text_file

# for building model
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules import FeedForward
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks import *

train_data_path = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/train.jsonl"
validation_data_path = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/dev.jsonl"
pretrained_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"


class PublicationDatasetReader(DatasetReader):
    ''' Dataset Reader for publication and venue dataset '''
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer]= None,**kwargs):
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        
    def _read(self, file_path: str) -> Iterator[Instance]:
        """ Read publication and venue dataset in JSON format in Lazy manner. It yields the generator
            Data is in the following format:
                {'title': ... 'paperAbstract': ... 'venue': ...}
        """
        instances = []
        with open(cached_path(file_path),'r') as data_file:
            for line in data_file:
                line = line.strip('\n')
                if not line:
                    continue
                paper_json = json.loads(line)
                title = paper_json['title']
                abstract = paper_json['paperAbstract']
                venue = paper_json['venue']
                
                yield self.text_to_instance(title, abstract, venue)
        
    def text_to_instance(self, 
                        title: str,
                        abstract: str,
                        venue: str = None)-> Instance:
        
            """ Turn title, abstract and venue to Instance """
            tokenized_title = self._tokenizer.tokenize(title)
            tokenized_abstract = self._tokenizer.tokenize(abstract)
            title_field = TextField(tokenized_title, self._token_indexers)
            abstract_field = TextField(tokenized_abstract, self._token_indexers)
            
            fields = {'title': title_field,
                        'abstract': abstract_field
                     }
            
            if venue is not None:
                fields['label'] = LabelField(venue)
            return Instance(fields)

class AcademicPaperClassifier(pl.LightningModule):
    """ Model to classify venue based on input title and abstract """
    def __init__(self, vocab, learning_rate=0.005, embedding_dim=100, hidden_dim= 100, batch_size=32, num_workers=8) ->None:
        super().__init__()
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.batch_size = batch_size
        self.num_workers= num_workers
        
        # reader
        self.reader = PublicationDatasetReader()
        
        # model will be created under create_model from `setup`
        num_classes = vocab.get_vocab_size('labels')
        vocab_length = vocab.get_vocab_size('tokens')
        
        token_embedding = Embedding(num_embeddings=vocab_length, 
                            embedding_dim=self.embedding_dim)
        
        self.text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
        self.title_encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(self.embedding_dim, self.hidden_dim, 
                                                 batch_first=True, bidirectional=True))
        self.abstract_encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(self.embedding_dim, self.hidden_dim, 
                                                    batch_first=True, bidirectional=True))
        
        self.classifier_feedforward = torch.nn.Linear(2 * 2 * self.hidden_dim, num_classes)

        self.loss = torch.nn.CrossEntropyLoss()
        
        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'accuracy3': CategoricalAccuracy(top_k=3)
        }
        self.save_hyperparameters()
        
    def prepare_data(self):
        self.train_data_path = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/train.jsonl"
        self.validation_data_path = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/dev.jsonl"
        self.pretrained_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"
        
    def setup1(self, stage=None):
        # create vocabulary
        
        train_dataset = self.reader.read(self.train_data_path)
        validation_dataset = self.reader.read(self.validation_data_path)
        
        
        # need to create vocabulary before
        self.vocab = Vocabulary.from_instances(train_dataset)
        self.vocab.extend_from_instances(validation_dataset)
        
    
    def train_dataloader(self):
        # use train dataset to create batches
        train_dl = MultiProcessDataLoader(self.reader, 
                                          data_path = self.train_data_path, 
                                          batch_size=self.batch_size, 
                                          shuffle=True,
                                         max_instances_in_memory=self.batch_size,
                                         num_workers=self.num_workers)
        train_dl.index_with(vocab)
        return train_dl
    
    def val_dataloader(self):
        data_loader = MultiProcessDataLoader(self.reader, self.validation_data_path, batch_size=self.batch_size, shuffle=False,
                                            max_instances_in_memory=self.batch_size,
                                            num_workers=self.num_workers)
        data_loader.index_with(vocab)
        return data_loader
        
    def forward(self,
               title: Dict[str, torch.LongTensor], 
               abstract: Dict[str, torch.LongTensor],
               label: torch.LongTensor = None)-> Dict[str, torch.Tensor]:
        
        embedded_title = self.text_field_embedder(title)
        title_mask = get_text_field_mask(title)
        
        encoded_title = self.title_encoder(embedded_title, title_mask)
        
        embedded_abstract = self.text_field_embedder(abstract)
        abstract_mask = get_text_field_mask(abstract)
        encoded_abstract = self.abstract_encoder(embedded_abstract, abstract_mask)
        logits = self.classifier_feedforward(torch.cat([encoded_title, encoded_abstract],dim=-1))
        class_probabilities = F.softmax(logits, dim=-1)
        argmax_indices = np.argmax(class_probabilities.cpu().data.numpy(), axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace='labels') for x in argmax_indices]
        output_dict = {
            'logits': logits,
            'class_prob': class_probabilities,
            'predicted_label': labels
        }
        
        if label is not None:
            loss = self.loss(logits, label)
            for name, metric in self.metrics.items():
                metric(logits, label)
                output_dict[name] = metric.get_metric()
                
            output_dict['loss'] = loss
            
        return output_dict
    
    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        return output
    def validation_step(self, batch, batch_idx):
        output = self.forward(**batch)
        output['val_loss'] = output['loss']
        del output['loss']
        return output
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        return parser

def build_vocab():

    # Building vocabulary
    reader = PublicationDatasetReader()
    train_dataset= reader.read(train_data_path)
    validation_dataset = reader.read(validation_data_path)
    vocab = Vocabulary.from_instances(train_dataset)
    vocab.extend_from_instances(validation_dataset)
    # vocabulary done

    return vocab

if __name__=='__main__':

    parser = ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = AcademicPaperClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # Logger
    logger = TensorBoardLogger('tb_logs', name='bert')

    # build vocab
    vocab = build_vocab()

    # model definition 
    model = AcademicPaperClassifier(vocab=vocab, batch_size=  args.batch_size, num_workers=args.num_workers)

    # Trainer definition
    trainer = pl.Trainer.from_argparse_args(args, fast_dev_run=False, progress_bar_refresh_rate=5, max_epochs=10,enable_pl_optimizer=False, 
                        callbacks=[
                            LogHistogramCallback(),
                            ModelCheckpoint(dirpath='.checkpoints/', monitor='val_loss')
                        ], logger=logger, auto_lr_find=True)

    trainer.fit(model)