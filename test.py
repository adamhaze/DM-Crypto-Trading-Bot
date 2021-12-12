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
import matplotlib.pyplot as plt

# warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

def add_label(df): 
    return ( ((df['Close'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)) * 100 )

# model params
num_classes = 3 
num_layers = 4
input_size = 14 # number of features
hidden_size = 500

neutral_thresh = 0.08 # % change threshold for buy/sell/hold

# set individual lag values 
lag_1min = 0
lag_5min = 2
lag_30min = 2
lag_1hr = 0
lag_4hr = 0
lag_12hr = 0
lag_24hr = 0

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




model = CryptoRNN(input_size, hidden_size, num_layers, num_classes).to(device)

model_saved = '../models/bitcoin_rnn_model'
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
days = 45
ignore_days = 3
labeled_data = df_5min.iloc[-int(288*days):-int(288*35),:]
# labeled_data = df_5min.iloc[-int(288*days):,:]
labeled_data['PercentChange'] = add_label(labeled_data)
labeled_data['label'] = np.where(labeled_data['PercentChange']> neutral_thresh, 1, 0)
labeled_data['label'] = np.where(labeled_data['PercentChange']< -neutral_thresh, 2,labeled_data['label'] )
del labeled_data['PercentChange']

print(labeled_data.head())

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
open_arr,close_arr = [],[]
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for X, Y in test_loader:
        open_arr.append(X[0][0][0].numpy())
        close_arr.append(X[0][0][3].numpy())
        y_true.append(Y.numpy())
        X,Y = X.to(device), Y.to(device)
        # X = torch.unsqueeze(X,1).float()
        X = X.type(torch.float)
        outputs = model(X)
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


##
## Model Profitability Evaluation
##
def eval_trading_strategy(pred, open, close):
    investment, shares = 0,0
    curr_money = 100000
    portfolio_tracker = [curr_money]

    print('Initial Investment: ${}'.format(curr_money))

    for i,elem in enumerate(pred):

        if elem == 0:
            if investment == 0: pass
            change = close[i+1] - open[i+1]
            investment += (shares * change)

        elif elem == 1:
            change = close[i+1] - open[i+1]
            if curr_money == 0:
                investment += (shares * change)
            else:
                shares = curr_money / close[i]
                investment = shares * close[i]

                investment += (shares * change)
                curr_money = 0

        elif elem == 2:
            if investment == 0: pass
            else:
                curr_money = (close[i] * shares)
                investment, shares = 0,0
        
        else: return 0

        tot = investment + curr_money
        portfolio_tracker.append(tot)
    print('Current Portfolio Value: ${}'.format(tot))

    return portfolio_tracker



portfolio_tracker = eval_trading_strategy(y_pred[:-1], open_arr, close_arr)

fig, ax = plt.subplots(figsize=(10,5))
x = [i for i in range(len(portfolio_tracker))]
ax.plot(x,portfolio_tracker)
ax.set_xlabel('Time Periods (5min)')
ax.set_ylabel('Investment Value ($)')
ax.set_title('Result of Model Trading Strategy with $100,000 initial investment')
# plt.legend()
fig.savefig('model_trading_strategy.png')
# """

"""
# function designed to take in real-time data and make real-time predictions
def predict_real_time(arr):

    current_point = arr[0]
    lag5_pt1 = arr[1]
    lag5_pt2 = arr[2]
    lag30_pt1 = arr[3]
    lag30_pt2 = arr[4]

    test_dataset = LiveData(current_point, lag5_pt1, lag5_pt2, lag30_pt1, lag30_pt2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for X, Y in test_loader:

        X,Y = X.to(device), Y.to(device)
        X = X.type(torch.float)
        outputs = model(X)

        _, predicted = torch.max(outputs.data, 1)
        return predicted
"""