import pytorch_lightning as pl
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import spacy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IWSLT2016
from torchtext.experimental.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

import numpy as np
import pandas as pd
import spacy
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import subprocess
from functools import partial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# create tokenizer
def tokenize_de(text,spacy_de):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text, spacy_en):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def generate_batch(data_batch, BOS_IDX, PAD_IDX, EOS_IDX):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=32):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download the tokenier
        def execute(command):
            lscommand = command.split()
            subprocess.run(lscommand)

        execute("python -m spacy download en_core_web_sm")
        execute("python -m spacy download de_core_news_sm")



    def setup(self, stage=None):
        # download, split etc.

        import de_core_news_sm
        import en_core_web_sm

        spacy_de = de_core_news_sm.load()
        spacy_en = en_core_web_sm.load()
        self.tokenizer = (partial(tokenize_de,spacy_de = spacy_de),
                        partial(tokenize_en, spacy_en = spacy_en)
                    )

        self.train_data, self.valid_data, self.test_data = Multi30k(root='.data', 
                                             split=('train', 'valid', 'test'), 
                                             language_pair=('de', 'en'),
                                             tokenizer= self.tokenizer
                                            )

        self.src_vocab, self.trg_vocab = self.train_data.get_vocab()

        # making sure that eos and pad ids are same. required to calcaulate
        # correct loss
        self.generate_batch = partial(generate_batch, BOS_IDX=self.src_vocab['<bos>'],
                                PAD_IDX = self.src_vocab['<pad>'],
                                EOS_IDX = self.src_vocab['<pad>'])


    def train_dataloader(self):
        # create data loaders here
        
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle=True, 
                collate_fn=self.generate_batch,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.batch_size, shuffle=False, 
                collate_fn = self.generate_batch,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.batch_size, shuffle=False, 
                collate_fn= self.generate_batch)

    
if __name__ == '__main__':
    dm = MyDataModule()
    dm.prepare_data()
    dm.setup()
    # check if we are able to itreate or not
    bs = next(iter(dm.train_dataloader()))
    print(bs[0].shape, bs[1].shape)