import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import torch
from torch.utils.data import DataLoader
import os,sys,glob
from Dataset import *
from models import *
from indicator_funcs import *
from sklearn.metrics import confusion_matrix
import warnings

def add_label(df): 
    return ( ((df['Open'].shift(-1) - df['Close'].shift(-1)) / df['Open'].shift(-1)) * 100 )

# model params
num_classes = 3 
num_layers = 2
input_size = 14 # number of features
hidden_size = input_size*2

neutral_thresh = 0.1 # % change threshold for buy/sell/hold

# set individual lag values 
lag_1min = 10
lag_5min = 0
lag_30min = 4
lag_1hr = 0
lag_4hr = 2
lag_12hr = 1
lag_24hr = 2

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

model = CryptoRNN(input_size, hidden_size, num_layers, num_classes).to(device)

model_saved = '../models/rnn_47%'
# model_saved = 'rnn_best_losses_checkpoint_v2'
model_state = torch.load(model_saved)
model.load_state_dict(model_state['state_dict'])


df_1min = pd.read_csv('BTC_Ticker_Data_1_Min.csv') 
df_5min = pd.read_csv('BTC_Ticker_Data_5_Min.csv') 
df_30min = pd.read_csv('BTC_Ticker_Data_30_Min.csv') 
df_1hr = pd.read_csv('BTC_Ticker_Data_1_Hour.csv') 
df_4hr = pd.read_csv('BTC_Ticker_Data_4_Hour.csv') 
df_12hr = pd.read_csv('BTC_Ticker_Data_12_Hour.csv') 
df_24hr = pd.read_csv('BTC_Ticker_Data_24_Hour.csv') 

# init test dataset
# Here we should collect live data for 1-2 hours or so
# create 1min, 5min, 30min, etc.. pandas dataframes that stop collecting once time ends
# next: push new data to SQL and compute indicators
# next: label new data in test.py
# only the new data gets passed into test_dataset and evaluated
# Note: for now we could just focus on collecting and labeling 5min interval data
days = 10
ignore_days = 3
labeled_data = df_5min.iloc[-int(288*days):,:]
labeled_data['PercentChange'] = add_label(labeled_data)
labeled_data['label'] = np.where(labeled_data['PercentChange']> neutral_thresh, 1, 0)
labeled_data['label'] = np.where(labeled_data['PercentChange']< -neutral_thresh, 2,labeled_data['label'] )
del labeled_data['PercentChange']

test_dataset = CryptoDataset(
    data_labeled = labeled_data,
    df_1min = df_1min.iloc[-int((days+ignore_days)*1440):,:],
    df_5min = df_5min.iloc[-int((days+ignore_days)*288):,:],
    df_30min = df_30min.iloc[-int((days+ignore_days)*48):,:],
    df_1hr = df_1hr.iloc[-int((days+ignore_days)*24):,:],
    df_4hr = df_4hr.iloc[-int((days+ignore_days)*6):,:],
    df_12hr = df_12hr.iloc[-int((days+ignore_days)*2):,:],
    df_24hr = df_24hr.iloc[-int((days+ignore_days)):,:],
    lag_1min = lag_1min,
    lag_5min = lag_5min,
    lag_30min = lag_30min,
    lag_1hr = lag_1hr,
    lag_4hr = lag_4hr,
    lag_12hr = lag_12hr,
    lag_24hr = lag_24hr,
    ignore_days = ignore_days
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print('~~~~~~~~~~~~ Testing Model Performance ~~~~~~~~~~~~')
y_true, y_pred = [],[]
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for X, Y in test_loader:
        y_true.append(Y.numpy())
        X,Y = X.to(device), Y.to(device)
        # X = torch.unsqueeze(X,1).float()
        X = X.type(torch.float)
        outputs = model.forward(X)
        # print(outputs)

        _, predicted = torch.max(outputs.data, 1)
        y_pred.append(predicted.numpy())
        # n_samples += Y.size(0)
        # print("Actual: {} | Predicted: {} ".format(Y,predicted))
        n_samples += 1
        if Y == predicted:
            n_correct += 1
        # n_correct += (predicted == Y).sum().item()

    acc = 100.0 * (n_correct / n_samples)
    print('\n')
    print(f'Accuracy: {acc} %')

y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)

print('\n')
print('Confusion Matrix: ')
cm = confusion_matrix(y_true,y_pred,labels=[0,1,2])
df_cm = pd.DataFrame(cm, index=[0,1,2], columns=[0,1,2])
print(df_cm)
print('\n')


"""
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
"""