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

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

neutral_thresh = 0.1
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

##
## Set all global parameters
##
path = 'data'
timeframe = 10
indicators = []
randomSeed = 41 # better way to do this: random number between 1 and 1 mill to ensure no 2 runs have same seed?
trainTestSplit = 0.8

batch_size = 128
num_epochs = 10
learning_rate = 1e-4
num_classes = 3 # buy / sell / hold -- more classes than this?
num_layers = 2
input_size = 5 + len(indicators) # number of features
hidden_size = 10

data = pd.read_csv('temp_data.csv', header=0).drop('Unnamed: 0',axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(np.array(data))
data_array = []
for i in range(len(data)):
    if i == len(data)-1: break
    ls = [k for k in data_normalized[i,:]]
    label = generate_label(data.iloc[i+1,3],data.iloc[i,3])
    ls.insert(0,label)
    data_array.append(ls)

features = ['label','Open','High','Low','Close','Volume']
df = pd.DataFrame(data_array, columns = features)
print(df.head())
mask = np.random.rand(len(df)) < trainTestSplit

df2 = df[mask]
valid_mask = np.random.rand(len(df2)) < 0.9

print('~~~~~~~~~~~~ Initializing Dataset ~~~~~~~~~~~~')
train_dataset = CryptoDataset(
    dataPath = df2[valid_mask],
    timeframe = timeframe,
    indicators = indicators,
    trainTestSplit = 0.8,
    seed = randomSeed
)
valid_dataset = CryptoDataset(
    dataPath = df2[~valid_mask],
    timeframe = timeframe,
    indicators = indicators,
    trainTestSplit = 0.8,
    seed = randomSeed,
    valid = True
)
######
test_dataset = CryptoDataset(
    dataPath = df[~mask],
    timeframe = timeframe,
    indicators = indicators,
    trainTestSplit = 0.5,
    seed = randomSeed+2,
    valid = False
)
######

print('Training Data: {} time frames'.format(len(train_dataset)))
# print('Validation Data: {} time frames'.format(len(valid_dataset)))

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
######
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
######


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
    # print(y)

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
        # optimizer.zero_grad()
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

# Save model after final training epoch
model_save = {'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
torch.save(model_save, 'rnn_result_v4')


##
## Testing
##
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for X, Y in test_loader:
        X,Y = X.to(device), Y.to(device)
        X = X.type(torch.float)
        outputs = model(X)

        _, predicted = torch.max(outputs.data, 1)
        n_samples += Y.size(0)
        n_correct += (predicted == Y).sum().item()

    acc = 100.0 * (n_correct / n_samples)
    print(f'Accuracy: {acc} %')
