# Deformation-Tracker
In this repo I'll explore diverse neural network architectures to predict the deformation of objects

Explanation of data:
There are three types of files.
1. Force file.This file has the following content:

To fix the code format according to the PEP8 standard:
```
black .
```

## Visualize training loss with Tensorboard
Tensorboard is used to visualize the training loss.

Run this command to run tensorboard:
```
tensorboard --logdir=./logs --port=6006
```

## Run the scripts
Run script with the default options
```
python src/basic_recurrent.py
```

See all the posible options you can execute the script with
```
python src/basic_recurrent.py --help
```

My resources:

To create a tf Dataset:
https://medium.com/when-i-work-data/converting-a-pandas-dataframe-into-a-tensorflow-dataset-752f3783c168

mypy type cheat sheet:
https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

simple encoder-decoder, sequence to sequence:
https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM#what-are-many-to-many-sequence-problems?


To investigate:

Furier Descriptors: https://arxiv.org/abs/1806.03857
paper: https://arxiv.org/pdf/1806.03857.pdf