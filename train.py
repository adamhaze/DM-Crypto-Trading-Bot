import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
import os,sys,glob
from Dataset import *
from models import *

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##
## Set all global parameters
##
path = 'data/'
timeframe = 10
indicators = []
randomSeed = 42 # better way to do this: random number between 1 and 1 mill to ensure no 2 runs have same seed?

batch_size = 100
num_epochs = 10
learning_rate = 1e-4
num_classes = 3 # buy / sell / hold -- more classes than this?
num_layers = 2

input_size = 4 + len(indicators) # number of features
hidden_size = 150


print('~~~~~~~~~~~~ Initializing Dataset ~~~~~~~~~~~~')
train_dataset = CryptoDataset(
    dataPath = path,
    timeframe = timeframe,
    indicators = indicators,
    trainTestSplit = 0.8,
    seed = randomSeed,
    train = True
)
valid_dataset = CryptoDataset(
    dataPath = path,
    timeframe = timeframe,
    indicators = indicators,
    trainTestSplit = 0.8,
    seed = randomSeed,
    train = False
)

# TODO: print some info about the dataset

print('~~~~~~~~~~~~ Initializing DataLoader ~~~~~~~~~~~~')
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False
)


print('~~~~~~~~~~~~ Initializing Model ~~~~~~~~~~~~')
model = CryptoRNN.CryptoRNN(input_size, hidden_size, num_layers, num_classes)

# Initialize loss critereron and gradient descent optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
