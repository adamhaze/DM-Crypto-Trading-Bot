import numpy as numpy
import pandas as pd
import torch


class CryptoDataset(Dataset):

	def __init__(self, timeframe, indicators, labels, train=False, live=False):

		self.train = train
		self.live = live
		self.tf = timeframe
		self.indicators = indicators
		self.labels = labels # compute labels in data loader?


	def __len__(self):
		return 0


	def __getitem__(self, idx):
		return 0