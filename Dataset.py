import numpy as numpy
import pandas as pd
import torch


class CryptoDataset(Dataset):

	def __init__(self, timeframe, labels, train=True):

		self.train = train
		self.tf = timeframe
		self.labels = labels # compute labels in data loader?