import pytorch_lightning as pl
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LogHistogramCallback(pl.Callback):
    def __init__(self, patience=25):
        self.patience = patience
        
    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.patience == 0:
            for k, v in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(tag=k, values=v.grad, global_step = trainer.global_step)

class ModelTestCallback(pl.Callback):
    def __init__(self, max_len=10, test ='puneet'):
        super().__init__()
        self.max_len = max_len
        self.test_sentence = test
    
    def on_fit_start(self, trainer, pl_module):
        # called when trainer setup is done.. model initiatlization has not happened yet
        self.transforms = pl_module.train_ds.transforms
        self.vocabs = pl_module.train_ds.get_vocab()
    
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        # take a random sentence and convert
        # apply src transforms on the text
        # here output contains the dictionary coming from training_step

        self.trg_vocab = self.vocabs[1]
        src_tensor = self.transforms[0](self.test_sentence).unsqueeze(0)
        src_mask = pl_module.make_src_mask(src_tensor)       # N X 1 X 6 X 6
        
        src_tensor = src_tensor.to(device)
        src_mask = src_mask.to(device)
        # output tensor
        out = ":"        # initial target
        out_tensor = self.transforms[1](out) 
        with torch.no_grad():
            enc_src = pl_module.model.encode(src_tensor, src_mask)
            
        trg_indices = [2]
        for i in range(self.max_len):
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
            trg_mask = pl_module.make_trg_mask(trg_tensor).to(device)
        
            with torch.no_grad():
                output = pl_module.model.decode(enc_src, src_mask, trg_tensor, trg_mask)
                output = pl_module.model.generator(output)

                pred_token = output.argmax(2)[:,-1].item()
                trg_indices.append(pred_token)
                
                if pred_token == PAD_IDX:
                    break

        trg_tokens = [self.trg_vocab.itos[i] for i in trg_indices]
        decode_string = ''.join(trg_tokens)
        print('decoded input : {} -> output: {} '.format(self.test_sentence, decode_string))
        trainer.logger.experiment.add_text('decodes', decode_string, trainer.current_epoch)
        
        return trg_tokens[1:]
        
        
