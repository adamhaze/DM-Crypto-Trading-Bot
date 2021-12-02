import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import torch
from torch.utils.data import DataLoader
import os,sys,glob
from Dataset import *
from models import *
from indicator_funcs import *


timeframe = 10
trainTestSplit = 0.8
num_classes = 3 # buy / sell / hold -- more classes than this?
num_layers = 2
input_size = 10 # number of features

batch_size = 128
num_epochs = 10
learning_rate = 5e-4
hidden_size = 10
lag = 3

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CryptoRNN(input_size, hidden_size, num_layers, num_classes).to(device)

model_saved = 'rnn_result_v5'
model_state = torch.load(model_saved)
model.load_state_dict(model_state['state_dict'])

# data = pd.read_csv('test_data.csv', header=1, usecols=desired_cols)
data = pd.read_csv('test_data.csv').drop('Unnamed: 0',axis=1)
# print(data.head())

# for i in range(len(data)):



# Need to call in real time data every timeframe
# add new row to local pandas dataframe
# re create a CryptoDataset instance and a DataLoader instance
# can't do this with less than lag time steps

test_data = CryptoDataset(data.iloc[100:105,:],timeframe,lag)
loader = DataLoader(test_data, batch_size=None, shuffle=False)

for X,Y in loader:
    X,Y = X.to(device), Y.to(device)
    X = X.type(torch.float)
    outputs = model(X)

    _, predicted = torch.max(outputs.data, 1)

# make array of predicted values
# live_df = dataframe of only new 5 minute increment data
def eval_trading_strategy(predicted_arr, live_df):

    for i,elem in enumerate(predicted):
        if elem == 0.0:
            # netural
            pass
        elif elem == 1.0:
            # buy
            pass
        else:
            # sell
            pass

        live_df.iloc[i,4] # get close price
