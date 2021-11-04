import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
import os,sys,glob
from Dataset import *
from Models import *


##
## Set all global parameters
##
path = 'data/'
tf = 10
indicators = []
randomSeed = 42 # better way to do this: random number between 1 and 1 mill to ensure no 2 runs have same seed?

batch_size = 8
num_epochs = 100
learning_rate = 1e-4



print('~~~~~~~~~~~~ Initializing Dataset ~~~~~~~~~~~~')
train_dataset = CryptoDataset(
    dataPath = path,
    timeframe = tf,
    indicators = indicators,
    trainTestSplit = 0.8,
    seed = randomSeed,
    train = True
)
valid_dataset = CryptoDataset(
    dataPath = path,
    timeframe = tf,
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

"""
print('~~~~~~~~~~~~ Initializing Model ~~~~~~~~~~~~')
model = CryptoRNN.CryptoRNN()

# Initialize loss critereron and gradient descent optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
"""