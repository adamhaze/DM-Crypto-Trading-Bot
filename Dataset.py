import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
from sklearn.preprocessing import MinMaxScaler


# TODO: set % price change thresholds for BUY, SELL, and HOLD conditions -- these should depend on timeframe
neutral_thresh = 0.1

# df = timeframe X+1 data
def generate_label2(df, prevClose):
	# compute % change in price from previous close
	curr_close = df.iloc[-1,4]
	change = curr_close - prevClose
	pct_change = (abs(change) / prevClose) * 100

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
class CryptoDataset(Dataset):

	def __init__(self, dataPath, timeframe, lag, valid=False, live=False):

		self.valid = valid
		self.live = live
		self.tf = timeframe
		self.lag = lag
		# self.scaler = MinMaxScaler(feature_range=(0, 1))
		# data = pd.read_csv(dataPath, header=0).drop('Unnamed: 0',axis=1)
		# data_normalized = self.scaler.fit_transform(np.array(data))
		
		# data_array = []
		# for i in range(len(data)):
		# 	if i == len(data)-1: break
		# 	ls = [k for k in data_normalized[i,:]]
		# 	label = generate_label(data.iloc[i+1,3],data.iloc[i,3])
		# 	ls.insert(0,label)
		# 	data_array.append(ls)


		# # TODO: add indicators to features and indicator values to candlestick_data above
		# features = ['label','Open','High','Low','Close','Volume']
		# df = pd.DataFrame(data_array, columns = features)

		# # train / test splitting
		# np.random.seed(seed)
		# mask = np.random.rand(len(df)) < trainTestSplit
		# cutoff = int(0.8 * len(df))
		# if self.valid:
		# 	self.dataFrame = df[~mask]
		# else:
		# 	self.dataFrame = df[mask]
		# if self.valid:
		# 	self.dataFrame = df.iloc[cutoff:,:]
		# else:
		# 	self.dataFrame = df.iloc[:cutoff,:]
		self.dataFrame = dataPath
		
	def __len__(self):
		return len(self.dataFrame)

	def __getitem__(self, idx):

		if idx < self.lag:
			candlestick = np.array(self.dataFrame.iloc[idx:idx+self.lag,1:])
		else:
			candlestick = np.array(self.dataFrame.iloc[idx-self.lag:idx,1:])
		candlestick_tensor = torch.from_numpy(candlestick)

		if self.live:
			return candlestick_tensor
		else:
			label = self.dataFrame.iloc[idx,0]
			return (candlestick_tensor, label)

	def get_labels(self):
		return self.dataFrame.iloc[:,0]
	def get_subset(self,start):
		return self.dataFrame.iloc[start:,:]



class CryptoDataset2(Dataset):

	def __init__(self, dataPath, timeframe, indicators, trainTestSplit, seed, valid=False, live=False):

		self.valid = valid
		self.live = live
		self.tf = timeframe
		self.indicators = indicators
		# data_files = glob.glob(dataPath + '/*.csv')
		# data_files = dataPath + '/gemini_BTCUSD_2020_1min.csv'
		# data_files = [data_files]
		data_files = [dataPath]
		self.scaler = MinMaxScaler(feature_range=(0, 1))

		data_array = []
		for fileNum, file in enumerate(data_files):
			desired_cols = ['Unix Timestamp', 'Open','High','Low','Close','Volume']
			data = pd.read_csv(file, header=1, usecols=desired_cols)[::-1] # flips data upside down
			num_time_chunks = int(data.shape[0] / timeframe)
			
			for i in range(num_time_chunks):
				candlestick_data = []
				start = i * self.tf
				end = start + self.tf
				query = data.iloc[start:end,:]
				# add first open price in current timeframe
				candlestick_data.append(query.iloc[0,1])
				# add max high value in current timeframe
				candlestick_data.append(query['High'].max())
				# add min low value in current timeframe
				candlestick_data.append(query['Low'].min())
				# add final close price for timeframe
				candlestick_data.append(query.iloc[-1,4])
				# add volume
				candlestick_data.append(query['Volume'].sum())
				# add label for current timeframe
				label = generate_label(data.iloc[start+self.tf:end+self.tf,:], query.iloc[-1,4])
				candlestick_data.insert(0,label)
				
				# append current timeframe data to main data array	
				data_array.append(candlestick_data)

		# TODO: add indicators to features and indicator values to candlestick_data above
		labs = np.array(data_array)[:,0].astype(int)
		data_normalized = self.scaler.fit_transform(np.array(data_array)[:,1:])
		data_final = np.insert(data_normalized,0,labs,axis=1)
		features = ['label','Open','High','Low','Close','Volume']
		df = pd.DataFrame(data_final, columns = features)

		# train / test splitting
		np.random.seed(seed)
		mask = np.random.rand(len(df)) < trainTestSplit
		cutoff = int(0.8 * len(df))
		if self.valid:
			self.dataFrame = df[~mask]
		else:
			self.dataFrame = df[mask]
		# if self.valid:
		# 	self.dataFrame = df.iloc[cutoff:,:]
		# else:
		# 	self.dataFrame = df.iloc[:cutoff,:]
		

			


	def __len__(self):
		return len(self.dataFrame)

	def __getitem__(self, idx):

		if idx < 5:
			candlestick = np.array(self.dataFrame.iloc[idx:idx+5,1:])
			# label = np.array(self.dataFrame.iloc[idx:idx+5,0])
		else:
			candlestick = np.array(self.dataFrame.iloc[idx-5:idx,1:])
			# label = np.array(self.dataFrame.iloc[idx-5:idx,0])
		candlestick_tensor = torch.from_numpy(candlestick)

		label = self.dataFrame.iloc[idx,0]
		return (candlestick_tensor, label)

	def get_labels(self):
		return self.dataFrame.iloc[:,0]
	def get_subset(self,start):
		return self.dataFrame.iloc[start:,:]
