import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import pytorch_lightning as pl

from data_utils import MyDataModule
from pytorch_lightning.loggers.neptune import NeptuneLogger

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'./')
from mllib.bert import *

MAX_EPOCHS=10
BATCHSIZE=128
LR=0.005
NUM_WORKERS=32

def create_logger()-> NeptuneLogger:
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGFmNDk0ODEtNzBmOC00YTY1LTkxZWQtM2Y1YzI5ZmRkMTY0In0=",
        project_name="puneetgirdhar.in/bert",
            close_after_fit=False,
        experiment_name="tutorial",  # Optional,
        params={"max_epochs": MAX_EPOCHS,
            "batch_size": BATCHSIZE,
            "lr": LR}, # Optional,
        tags=["pytorch-lightning", "mlp"],
        upload_source_files=['*.py','*.yaml'],
        upload_stderr=False,
        upload_stdout=False
        )

    return neptune_logger



class LITTransformer(pl.LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate=0.005, pad_idx=1):
        # get all parameters from config and make it part of class
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 256
        self.enc_layers = 3
        self.dec_layers = 3
        self.enc_heads =  8
        self.dec_heads =  8
        self.enc_pf_dim =  512
        self.dec_pf_dim =  512
        self.enc_dropout = 0.1
        self.dec_dropout = 0.1

        self.src_pad_idx = pad_idx
        self.trg_pad_idx = pad_idx
        self.learning_rate = learning_rate

        self.model = Transformer(self.input_dim, self.output_dim, self.src_pad_idx, self.trg_pad_idx)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)
        self.initialize_weights()

        self.save_hyperparameters()

    def dummy_run(self):
        ''' Test the functionality of the model '''
        x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
        trg = torch.tensor([[1,7,4,3,5,9,2,0],[1,5,6,2,4,7,6,2]]).to(device)

        src_pad_idx = 0
        trg_pad_idx = 0
        src_vocab_size = 10
        trg_vocab_size= 10

        model = self.model(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)
        print(model(x, trg[:,:-1])[0].shape)


    def initialize_weights(self):
        def apply_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
            
        self.model.apply(apply_weights)

    def training_step(self, batch, batch_idx):
        ''' Training step '''
        src = batch[0].permute(1, 0)
        trg = batch[1].permute(1, 0)
        output, _ = self.model(src, trg)


        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim) #output = [batch size * trg len - 1, output dim]
        trg = trg.contiguous().view(-1) #trg = [batch size * trg len - 1]
        
        loss = self.criterion(output, trg)
        self.log('loss',loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        ret = self.training_step(batch, batch_idx)
        self.log('val_loss', ret['loss'])
        return {'val_loss': ret['loss']}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

if __name__=='__main__':
    dm = MyDataModule(batch_size = BATCHSIZE, num_workers=NUM_WORKERS)

    model = LITTransformer(input_dim=19213, output_dim=10836,
                    learning_rate=LR)
    trainer = pl.Trainer(gpus=2, logger=create_logger(),max_epochs=MAX_EPOCHS,
                            accelerator='ddp', auto_lr_find=True)

    #trainer = pl.Trainer(gpus=1, logger=create_logger(),max_epochs=MAX_EPOCHS,auto_lr_find=True)

    trainer.fit(model, dm)
    trainer.save_checkpoint('trainer.ckpt')
