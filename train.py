import warnings
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import torch
from torch.utils.data import DataLoader
from torch import optim
import os,sys,glob
from Dataset import *
from models import *
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from indicator_funcs import *
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set output directory for model checkpoints and loss plots
# output_directory = 'output/'

neutral_thresh = 0.2
def generate_label(nextClose, currClose):
	# compute % change in price from previous close
	change = nextClose - currClose
	pct_change = (abs(change) / currClose) * 100

	# conditons for BUY/SELL/HOLD based on % price change
	if pct_change <= neutral_thresh:
		# % change in price insignificant enough to indicate HOLD
		return 1
	elif change < 0:
		# negative % change in price indicating SELL
		return 0
	else:
		# positive % change in price indicating BUY
		return 2

def compute_ind_and_label(df):

    dat = np.array(df)
    data_array = []
    for i in range(50,len(dat)):
        if i == len(dat)-1: break
        ls = [k for k in dat[i,:]]
        rsi = RSI(df.iloc[i-14:i,:])[-1]
        ls.append(rsi)
        ma5 = MA(df.iloc[i-5:i,:])
        ma8 = MA(df.iloc[i-8:i,:])
        ma13 = MA(df.iloc[i-13:i,:])
        ma20 = MA(df.iloc[i-20:i,:])
        ma50 = MA(df.iloc[i-50:i,:])
        ls.append(ma5)
        ls.append(ma8)
        ls.append(ma13)
        ls.append(ma20)
        ls.append(ma50)
        # macd = MACD(dat.iloc[i-26:i,:])

        label = generate_label(df.iloc[i+1,3],df.iloc[i,3])
        ls.insert(0,label)
        data_array.append(ls)

    features = ['label','Open','High','Low','Close','Volume','RSI','MA_5','MA_8','MA_13','MA_20','MA_50']
    df_final = pd.DataFrame(data_array, columns = features)
    df_final.to_csv('temp_data_30min_allfeats.csv')
    return df_final

def save_losses():
    # plt.ylim([0,1])
    plt.plot(train_losses, label='training loss')
    plt.plot(valid_losses, label='validation loss')
    plt.xlabel('Training epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig('losses.png')
##
## Set all global parameters
##
trainTestSplit = 0.9
num_classes = 3 # buy / sell / hold -- more classes than this?
num_layers = 1
input_size = 14 # number of features

batch_size = 1024
num_epochs = 1
learning_rate = 1e-1
hidden_size = input_size

# set individual lag values
lag_1min = 5
lag_5min = 0
lag_30min = 1
lag_1hr = 0
lag_4hr = 0
lag_12hr = 0
lag_24hr = 0

print('batch size: {}'.format(batch_size))
print('epochs: {}'.format(num_epochs))
print('lr: {}'.format(learning_rate))
print('num_layers: {}'.format(num_layers))
print('hidden_size: {}'.format(hidden_size))
print('neutral thresh: {}'.format(neutral_thresh))
print('1 minute lag: {}'.format(lag_1min))
print('5 minute lag: {}'.format(lag_5min))
print('30 minute lag: {}'.format(lag_30min))
print('1 hour lag: {}'.format(lag_1hr))
print('4 hour lag: {}'.format(lag_4hr))
print('12 hour lag: {}'.format(lag_12hr))
print('24 hour lag: {}'.format(lag_24hr))

# Load ENV Files for SQL database
env_vars = {} # or dict {}
with open('src/env.txt') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        key, value = line.strip().split('=', 1)
        env_vars[key]= value
print(env_vars)

# create sqlalchemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                       .format(user=env_vars['USER'],
                               pw=env_vars['PASSWORD'],
                               host=env_vars['HOST'],
                               db=env_vars['DB']))
cols = ['index','Unix Timestamp','Date','Symbol','Open','High','Low','Close','Volume']

# pull timeframe data
df_1min = pd.read_sql_table('BTC_Ticker_Data', con=engine, index_col='index').drop(['Date','Symbol'],axis=1)
df_5min = pd.read_sql_table('BTC_Ticker_Data_5_Min', con=engine, index_col='index').drop(['Date','Symbol','Time Frame'],axis=1)
df_30min = pd.read_sql_table('BTC_Ticker_Data_30_Min', con=engine, index_col='index').drop(['Date','Symbol','Time Frame'],axis=1)
df_1hr = pd.read_sql_table('BTC_Ticker_Data_1_Hour', con=engine, index_col='index').drop(['Date','Symbol','Time Frame'],axis=1)
df_4hr = pd.read_sql_table('BTC_Ticker_Data_4_Hour', con=engine, index_col='index').drop(['Date','Symbol','Time Frame'],axis=1)
df_12hr = pd.read_sql_table('BTC_Ticker_Data_12_Hour', con=engine, index_col='index').drop(['Date','Symbol','Time Frame'],axis=1)
df_24hr = pd.read_sql_table('BTC_Ticker_Data_24_Hour', con=engine, index_col='index').drop(['Date','Symbol','Time Frame'],axis=1)

print('1 minute: ', df_1min.columns)
print('5 minute: ', df_5min.columns)

# TODO: ************ LABELING ************
labeled_data = df_5min # replace this with labeled data

labeled_data = labeled_data.iloc[(864*675):,:] # 864 = 3 days
mask = np.random.rand(len(labeled_data)) < trainTestSplit
labeled_data_masked = labeled_data[mask]
valid_mask = np.random.rand(len(labeled_data_masked)) < 0.9


print('~~~~~~~~~~~~ Initializing Dataset ~~~~~~~~~~~~')
train_dataset = CryptoDataset(
    data_labeled = labeled_data_masked[valid_mask],
    df_1min = df_1min,
    df_5min = df_5min,
    df_30min = df_30min,
    df_1hr = df_1hr,
    df_4hr = df_4hr,
    df_12hr = df_12hr,
    df_24hr = df_24hr,
    lag_1min = lag_1min,
    lag_5min = lag_5min,
    lag_30min = lag_30min,
    lag_1hr = lag_1hr,
    lag_4hr = lag_4hr,
    lag_12hr = lag_12hr,
    lag_24hr = lag_24hr
)
valid_dataset = CryptoDataset(
    data_labeled = labeled_data_masked[~valid_mask],
    df_1min = df_1min,
    df_5min = df_5min,
    df_30min = df_30min,
    df_1hr = df_1hr,
    df_4hr = df_4hr,
    df_12hr = df_12hr,
    df_24hr = df_24hr,
    lag_1min = lag_1min,
    lag_5min = lag_5min,
    lag_30min = lag_30min,
    lag_1hr = lag_1hr,
    lag_4hr = lag_4hr,
    lag_12hr = lag_12hr,
    lag_24hr = lag_24hr
)
test_dataset = CryptoDataset(
    data_labeled = labeled_data[~mask],
    df_1min = df_1min,
    df_5min = df_5min,
    df_30min = df_30min,
    df_1hr = df_1hr,
    df_4hr = df_4hr,
    df_12hr = df_12hr,
    df_24hr = df_24hr,
    lag_1min = lag_1min,
    lag_5min = lag_5min,
    lag_30min = lag_30min,
    lag_1hr = lag_1hr,
    lag_4hr = lag_4hr,
    lag_12hr = lag_12hr,
    lag_24hr = lag_24hr
)

print('Training Data: {} time frames'.format(len(train_dataset)))
print('Validation Data: {} time frames'.format(len(valid_dataset)))

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
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


print('~~~~~~~~~~~~ Initializing Model ~~~~~~~~~~~~')
model = CryptoRNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)

# Initialize loss critereron and gradient descent optimizer
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    y1 = train_dataset.get_labels().to_list()
    y2 = valid_dataset.get_labels().to_list()
    y = y1 + y2
    print('Class 0: {}'.format(y.count(0.0)))
    print('Class 1: {}'.format(y.count(1.0)))
    print('Class 2: {}'.format(y.count(2.0)))

    class_wts = compute_class_weight('balanced',np.unique(y),y)
    class_wts = torch.from_numpy(class_wts).float()
    criterion = torch.nn.CrossEntropyLoss(weight=class_wts)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)


print('~~~~~~~~~~~~ Training Model ~~~~~~~~~~~~')
train_losses, valid_losses = [],[]

for epoch in range(num_epochs):

    print("EPOCH: {} ".format(epoch),end='',flush=True)

    sum_loss = 0
    for batch, (X,Y) in enumerate(train_loader):

        X,Y = X.to(device), Y.to(device)
        # X = torch.unsqueeze(X,1).float()
        X = X.type(torch.float)
        Y = Y.type(torch.LongTensor)

        # Compute forward pass
        Y_hat = model.forward(X).to(device)

        # Calculate training loss
        optimizer.zero_grad()
        loss = criterion(Y_hat, Y)

        # Perform backprop and zero gradient
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()

        sum_loss = sum_loss + criterion(Y_hat, Y)
    
    train_losses.append(sum_loss.item()/batch)

    #Valid
    loss = 0
    for batch, (X,Y) in enumerate(valid_loader):

        X,Y = X.to(device),Y.to(device)
        X = X.type(torch.float)
        Y = Y.type(torch.LongTensor)

        # Compute forward pass
        Y_hat = model.forward(X).to(device)

        # Calculate training loss
        loss = loss + criterion(Y_hat, Y)

    valid_losses.append(loss.item()/batch)
    print("\tTRAIN LOSS = {:.5f}\tVALID LOSS = {:.5f}".format(train_losses[-1],valid_losses[-1]))
    # print("\tTRAIN LOSS = {:.5f}".format(train_losses[-1]))

    if valid_losses[-1] == np.array(valid_losses).min():
        checkpoint = {'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
        # torch.save(checkpoint, 'checkpoint_epoch_{}'.format(epoch))
        torch.save(checkpoint, 'rnn_best_losses_checkpoint')
    else:
        if len(valid_losses) > np.array(valid_losses).argmin() + 50:
            print('No improvement in last 50 epochs... terminating...')
            break

# Save model after final training epoch
# model_save = {'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
# torch.save(model_save, 'rnn_result_v5')
# save_losses()

##
## Testing
##
y_true, y_pred = [],[]
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for X, Y in test_loader:
        y_true.append(Y.numpy())
        X,Y = X.to(device), Y.to(device)
        X = X.type(torch.float)
        outputs = model(X)

        _, predicted = torch.max(outputs.data, 1)
        y_pred.append(predicted.numpy())
        n_samples += Y.size(0)
        n_correct += (predicted == Y).sum().item()

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
