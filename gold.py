# -*- coding: utf-8 -*-
"""
list prompt example
"""
from __future__ import print_function, unicode_literals

from pprint import pprint

from PyInquirer import prompt, Separator

from examples import custom_style_2
import os


def get_delivery_options(answers):
    options = ['Not Optimized', 'Grid Search - Cross Validation']
    # if answers['size'] == 'jumbo':
    #     options.append('helicopter')
    return options


questions = [
    {
        'type': 'list',
        'name': 'model',
        'message': 'Choose Prediction Model: ',
        'choices': [
            'LSTM',
            'GRU'
        ]
    },
    {
        'type': 'list',
        'name': 'size',
        'message': 'Choose Train Data Time Length:',
        'choices': [
            'data/train-90.csv',
            'data/train-120.csv',
            'data/train-180.csv',
            'data/train-360.csv',
            'data/train-1080.csv',
            'data/train-1800.csv',
            'data/train-3600.csv',
            'data/train.csv',
            'data/train-90-validation.csv',
            'data/train-120-validation.csv',
            'data/train-180-validation.csv',
            'data/train-360-validation.csv',
            'data/train-1080-validation.csv',
            'data/train-1800-validation.csv',
            'data/train-3600-validation.csv',
            'data/train-validation.csv'
        ],
        'filter': lambda val: val.lower()
    },
    {
        'type': 'list',
        'name': 'initializer',
        'message': 'Kernel Initializer: ',
        'choices': [
            'lecun_uniform',
            'zero',
            'ones',
            'glorot_normal',
            'glorot_uniform',
            'he_normal',
            'he_uniform',
            'uniform',
            'normal',
            'orthogonal',
            'constant',
            'random_normal',
            'random_uniform'
        ]
    },
    {
        'type': 'list',
        'name': 'batch',
        'message': 'Batch Size: ',
        'choices': [
            '16',
            '32',
            '64',
            '128',
            '256',
            '512',
            '1024'
        ]
    },
    {
        'type': 'list',
        'name': 'dropout',
        'message': 'Dropout Rate: ',
        'choices': [
            '0.0',
            '0.2',
            '0.3',
            '0.4'
        ]
    },
    {
        'type': 'list',
        'name': 'units',
        'message': 'Neuron Units: ',
        'choices': [
            '32',
            '64',
            '128'
        ]
    },
    {
        'type': 'list',
        'name': 'optimizer',
        'message': 'Learning Optimizer: ',
        'choices': [
            'SGD',
            'RMSProp',
            'Adagrad',
            'Adam'
        ]
    },
    {
        'type': 'list',
        'name': 'epoch',
        'message': 'Epochs: ',
        'choices': [
            '25',
            '50',
            '100',
            '200'
        ]
    }
    # {
    #     'type': 'list',
    #     'name': 'tuning',
    #     'message': 'Optimized with Hyperparameter Tuning?',
    #     'choices': get_delivery_options,
    # },
]

answers = prompt(questions, style=custom_style_2)
timesize = answers['size'].replace(' ', '-')

DATA_SOURCE = answers['size']
KERNEL_INITIALIZER = answers['initializer']
BATCH_SIZE = answers['batch']
DROPOUT_RATE = answers['dropout']
NEURON_UNITS = answers['units']
LEARNING_OPTIMIZER = answers['optimizer']
EPOCH = answers['epoch']

if answers['model'] == "GRU":
    os.system(f'python3 gru.py '
              f'{DATA_SOURCE} '
              f'{KERNEL_INITIALIZER} '
              f'{BATCH_SIZE} '
              f'{DROPOUT_RATE} '
              f'{NEURON_UNITS} '
              f'{LEARNING_OPTIMIZER} '
              f'{EPOCH} '
              )
else:
    os.system(f'python3 lstm.py '
              f'{DATA_SOURCE} '
              f'{KERNEL_INITIALIZER} '
              f'{BATCH_SIZE} '
              f'{DROPOUT_RATE} '
              f'{NEURON_UNITS} '
              f'{LEARNING_OPTIMIZER} '
              f'{EPOCH} '
              )

# execute = execute.lower()
# pprint('Running ' + execute)
# exec(open(execute).read())
