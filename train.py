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
path = 'data'
timeframe = 10
indicators = []
randomSeed = 42 # better way to do this: random number between 1 and 1 mill to ensure no 2 runs have same seed?

batch_size = 100
num_epochs = 10
learning_rate = 1e-4
num_classes = 3 # buy / sell / hold -- more classes than this?
num_layers = 2

input_size = 5 + len(indicators) # number of features
hidden_size = 15


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
    train = True,
    valid = True
)

print('Training Data: {} time frames'.format(len(train_dataset)))
print('Validation Data: {} time frames'.format(len(valid_dataset)))

# TODO: print some info about the dataset

print('~~~~~~~~~~~~ Initializing DataLoader ~~~~~~~~~~~~')
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False
)


print('~~~~~~~~~~~~ Initializing Model ~~~~~~~~~~~~')
model = CryptoRNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)

# Initialize loss critereron and gradient descent optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


print('~~~~~~~~~~~~ Training Model ~~~~~~~~~~~~')
train_losses, valid_losses = [],[]

for epoch in range(num_epochs):

    print("EPOCH: {} ".format(epoch),end='',flush=True)

    sum_loss = 0
    for batch, (X,Y) in enumerate(train_loader):

        X,Y = X.to(device), Y.to(device)
        X = torch.unsqueeze(X,1).float()

        # Compute forward pass
        Y_hat = model.forward(X).to(device)

        # Calculate training loss
        loss = criterion(Y_hat, Y)

        # Perform backprop and zero gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()

        sum_loss = sum_loss + criterion(Y_hat, Y)
    
    train_losses.append(sum_loss.item()/batch)

    #Valid
    loss = 0
    for batch, (X,Y) in enumerate(valid_loader):

        X,Y = X.to(device),Y.to(device)
        X = torch.unsqueeze(X,1).float()

        # Compute forward pass
        Y_hat = model.forward(X).to(device)

        # Calculate training loss
        loss = loss + criterion(Y_hat, Y)

    valid_losses.append(loss.item()/batch)
    print("\tTRAIN LOSS = {:.5f}\tVALID LOSS = {:.5f}".format(train_losses[-1],valid_losses[-1]))

