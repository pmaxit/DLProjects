import sys
sys.path.insert(0,'./')

from argparse import ArgumentParser
from mllib.new_bert import *
from data_utils import *
from callbacks import *

import string
import random
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD

from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader,random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import subprocess
from functools import partial
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl

PAD_IDX=  1

class LitTransformer(pl.LightningModule):
    def __init__(self, learning_rate=0.001, batch_size=4, num_workers=0):
        super().__init__()
        self.learning_rate=learning_rate
        self.batch_size = batch_size
        self.num_workers=num_workers
        
        self.loss_crit = LabelSmoothingLoss2(ignore_value = 1, label_smoothing=0.1)
        self.save_hyperparameters()

        
    def make_src_mask(self, src):
        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        # (N , 1, 1, src_len)
        return src_mask
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N , 1, trg_len, trg_len)
        return trg_mask
    
    def forward(self, src, trg):
        
        # get mask for src
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        return self.model.forward(src, src_mask, trg, trg_mask)
        
        
    def prepare_data(self):
        data = [[x,y] for x, y in random_examples(10000,10)]
        self.raw_data={}
        self.raw_data['train'], self.raw_data['test'] = train_test_split(data, test_size=0.33, random_state = 42)
        
    
    def setup(self, stage = None):
        tokenizer = list, list
        reversed_train, reversed_test = ReversedString(data = self.raw_data, tokenizer=tokenizer)
        
        # save the vocab
        self.src_vocab, self.trg_vocab = reversed_train.get_vocab()
        
        # define the model based on trg vocab. Note: We don't use src_vocab here.
        self.model = make_model(len(self.trg_vocab), len(self.trg_vocab), 
                               N=4, d_model=128, d_ff=128, h=4, dropout=0.2)
        
        self.criterion = SimpleLossCompute(self.model.generator, self.loss_crit, None)

        # train / val split
        n = len(reversed_train)
        p = int(0.8*n)
        rerversed_train, reversed_val = random_split(reversed_train, [p, n-p])
        
        # asssign to use in dataloaders
        self.train_ds = reversed_train
        self.test_ds = reversed_test
        self.val_ds = reversed_val
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
        # scheduler_warmup is chained with schduler_steplr
        scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
        scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
    
        return [optim],[scheduler_warmup]
        
    
    def training_step(self, batch, batch_idx):
        src, trg = batch
        src = src.permute(1,0)
        trg = trg.permute(1,0)
        
        # pass through seq2seq model and get loss
        out =  self.forward(src,trg[:,:-1])
        loss = self.criterion(out, trg[:,1:])
        self.log('loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        ret = self.training_step(batch, batch_idx)
        self.log('val_loss', ret['loss'])
        return {'val_loss': ret['loss']}
        
    def train_dataloader(self):
        dl = DataLoader(self.train_ds, self.batch_size,
                          collate_fn=generate_batch_new, num_workers=self.num_workers)
        return dl
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size,
                          collate_fn=generate_batch_new,num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size,
                          collate_fn=generate_batch_new,num_workers=self.num_workers)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        return parser

if __name__ == '__main__':
    parser = ArgumentParser()
    

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTransformer.add_model_specific_args(parser)
    args = parser.parse_args()

    logger = TensorBoardLogger('tb_logs', name='bert')
    model = LitTransformer(batch_size=args.batch_size, num_workers=args.num_workers, learning_rate=args.learning_rate)
    
    trainer = pl.Trainer.from_argparse_args(args, fast_dev_run=False, progress_bar_refresh_rate=5, max_epochs=10,enable_pl_optimizer=False, 
                        callbacks=[
                            ModelTestCallback(test='puneet'), 
                            LogHistogramCallback(),
                            ModelCheckpoint(dirpath='.checkpoints/', monitor='val_loss')
                        ], logger=logger, auto_lr_find=True)

    trainer.fit(model)
