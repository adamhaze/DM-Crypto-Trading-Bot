import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob

desired_cols = ['Unix Timestamp', 'Date', 'Open','High','Low','Close']
dat = pd.read_csv('data/gemini_BTCUSD_2021_1min.csv', header=1, usecols=desired_cols)[::-1]
# dat = dat[::-1]
# print(dat.iloc[0,2])
d = dat.iloc[0:10,:]
print(d.iloc[-1,5] - 5)

# TODO: set % price change thresholds for BUY, SELL, and HOLD conditions -- these should depend on timeframe
neutral_thresh = 1.0

# df = timeframe X+1 data
def generate_label(df, prevClose):
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

class CryptoDataset(Dataset):

	def __init__(self, dataPath, timeframe, indicators, trainTestSplit, seed, train=False, live=False):

		self.train = train
		self.live = live
		self.tf = timeframe
		self.indicators = indicators
		data_files = glob.glob(dataPath + '/*.csv')

		data_array = []
		for fileNum, file in enumerate(data_files):
			desired_cols = ['Unix Timestamp', 'Open','High','Low','Close']
			data = pd.read_csv(file, header=1, usecols=desired_cols)[::-1]
			num_time_chunks = int(data.shape[0] / timeframe)
			
			candlestick_data = []
			# TODO: handle missing chunks of time
			for i in range(num_time_chunks):
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
				# add label for current timeframe
				label = generate_label(data.iloc[start+self.tf:end+self.tf,:], query.iloc[-1,4])
				candlestick_data.insert(0,label)
				
			# append current timeframe data to main data array	
			data_array.append(candlestick_data)


		# TODO: add indicators to features and indicator values to candlestick_data above
		features = ['label','Open','High','Low','Close']
		df = pd.DataFrame(data_array, columns = features)

		# train / test splitting
		np.random.seed(seed)
		mask = np.random.rand(len(df)) < trainTestSplit
		if self.train:
			self.dataFrame = df[mask]
		else:
			self.dataFrame = df[~mask]
			


	def __len__(self):
		return len(self.dataFrame)

	def __getitem__(self, idx):

		candlestick = self.dataFrame.iloc[idx,1:]
		candlestick_tensor = torch.from_numpy(candlestick)

		if self.train:
			label = self.dataFrame.iloc[idx,0]
			return (candlestick_tensor, label)
		else:
			return candlestick_tensor
