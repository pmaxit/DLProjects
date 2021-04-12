# Name Generation
> Name generation using LSTM model


This project aims to generate names automatically by reading initial characters.

## Install

`pip install mllib`

## How to use

To generate names randomly, we first need to load the model and call `gen_name` function.

## Pytorch Lightning grid
1. grid interactive create --g_name bert-transformer
2. grid interactive ssh bert-transformer


```python
# get the trainer from the module
model = namegen.get_first_name_model()
model.cuda() # TODO remove this requirement
```




    RNN(
      (dropout): Dropout(p=0.2, inplace=False)
      (embedding): Embedding(104, 30, scale_grad_by_freq=True)
      (rnn): LSTM(30, 300, num_layers=2, batch_first=True, dropout=0.2)
      (decoder): Linear(in_features=300, out_features=104, bias=True)
      (criterion): CrossEntropyLoss()
    )



```python
model.generate("CHRIS")
```




    'CHRISTOPHER'



# End.
