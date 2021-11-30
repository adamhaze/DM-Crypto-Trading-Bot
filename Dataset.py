import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
from sklearn.preprocessing import MinMaxScaler


class CryptoDataset(Dataset):

	def __init__(self, data_labeled, df_1min, df_5min, df_30min, df_1hr, df_4hr, df_12hr, df_24hr, lag_1min=0, lag_5min=0, lag_30min=0, lag_1hr=0, lag_4hr=0, lag_12hr=0, lag_24hr=0, valid=False, live=False):

		self.df_1min = df_1min
		self.df_5min = df_5min
		self.df_30min = df_30min
		self.df_1hr = df_1hr
		self.df_4hr = df_4hr
		self.df_12hr = df_12hr
		self.df_24hr = df_24hr

		self.lag_1min = lag_1min
		self.lag_5min = lag_5min
		self.lag_30min = lag_30min
		self.lag_1hr = lag_1hr
		self.lag_4hr = lag_4hr
		self.lag_12hr = lag_12hr
		self.lag_24hr = lag_24hr

		self.valid = valid
		self.live = live
		self.features = ['Unix Timestamp','Open','High','Low','Close','Volume','MA_5','MA_8','MA_10','MA_13','MA_20','MA_50','RSI','MACD','M_Signal']
		self.lag_open = 864 # num time points to skip from beginning of data: 864 = 3 days in 5min timeframe

		self.dataFrame = data_labeled
		labs = [0. for i in range(len(self.dataFrame))]
		labs[0] = 1.
		labs[1] = 2.
		self.dataFrame.insert(len(self.features),'label',labs)

	def __len__(self):
		return len(self.dataFrame)

	def __getitem__(self, idx):

		current_unix = self.dataFrame.iloc[idx,0]
		relevant_data_points = pd.DataFrame([],columns=self.features)
		relevant_data_points = relevant_data_points.append(self.dataFrame.iloc[idx,:len(self.features)])

		if self.lag_1min != 0:
			idx_1min = int((idx + self.lag_open) * 5)
			relevant_data_points = relevant_data_points.append(self.df_1min.iloc[idx_1min-self.lag_1min:idx_1min])
			# temp_df = self.df_1min[self.df_1min['Unix Timestamp'] <= current_unix].iloc[-self.lag_1min:]
			# relevant_data_points = relevant_data_points.append(temp_df)
		if self.lag_5min != 0:
			idx_5min = idx+self.lag_open
			relevant_data_points = relevant_data_points.append(self.df_5min.iloc[idx_5min-self.lag_5min:idx_5min])
			# temp_df = self.df_5min[self.df_5min['Unix Timestamp'] <= current_unix].iloc[-self.lag_5min:]
			# relevant_data_points = relevant_data_points.append(temp_df)
		if self.lag_30min != 0:
			idx_30min = int((self.lag_open+idx)/6)
			relevant_data_points = relevant_data_points.append(self.df_30min.iloc[idx_30min-self.lag_30min:idx_30min])
			# temp_df = self.df_30min[self.df_30min['Unix Timestamp'] <= current_unix].iloc[-self.lag_30min:]
			# relevant_data_points = relevant_data_points.append(temp_df)
		if self.lag_1hr != 0:
			idx_1hr = int((idx + self.lag_open) / 12)
			relevant_data_points = relevant_data_points.append(self.df_1hr.iloc[idx_1hr-self.lag_1hr:idx_1hr])
			# temp_df = self.df_1hr[self.df_1hr['Unix Timestamp'] <= current_unix].iloc[-self.lag_1hr:]
			# relevant_data_points = relevant_data_points.append(temp_df)
		if self.lag_4hr != 0:
			idx_4hr = int((idx + self.lag_open) / 48)
			relevant_data_points = relevant_data_points.append(self.df_4hr.iloc[idx_4hr-self.lag_4hr:idx_4hr])
			# temp_df = self.df_4hr[self.df_4hr['Unix Timestamp'] <= current_unix].iloc[-self.lag_4hr:]
			# relevant_data_points = relevant_data_points.append(temp_df)
		if self.lag_12hr != 0:
			idx_12hr = int((idx + self.lag_open) / 144)
			relevant_data_points = relevant_data_points.append(self.df_12hr.iloc[idx_12hr-self.lag_12hr:idx_12hr])
			# temp_df = self.df_12hr[self.df_12hr['Unix Timestamp'] <= current_unix].iloc[-self.lag_12hr:]
			# relevant_data_points = relevant_data_points.append(temp_df)
		if self.lag_24hr != 0:
			idx_24hr = int((idx + self.lag_open) / 288)
			relevant_data_points = relevant_data_points.append(self.df_24hr.iloc[idx_24hr-self.lag_24hr:idx_24hr])
			# temp_df = self.df_24hr[self.df_24hr['Unix Timestamp'] <= current_unix].iloc[-self.lag_24hr:]
			# relevant_data_points = relevant_data_points.append(temp_df)

		relevant_data_points = relevant_data_points.drop(['Unix Timestamp'], axis=1)
		# relevant_data = pd.concat(ls).drop(['Unix Timestamp'], axis=1)
		data_tensor = torch.from_numpy(np.array(relevant_data_points, dtype=float))

		label = self.dataFrame.iloc[idx,-1]
		return (data_tensor, label)

	def get_labels(self):
		return self.dataFrame.iloc[:,-1]
	def get_subset(self,start):
		return self.dataFrame.iloc[start:,:]


