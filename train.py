import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
import os,sys,glob
from Dataset import *
from Models import *


path = 'data/'
tf = 10
indicators = []
randomSeed = 42 # better way to do this: random number between 1 and 1 mill to ensure no 2 runs have same seed?



print('~~~~~~~~~~~~ Initializing Dataset ~~~~~~~~~~~~')
train_dataset = CryptoDataset(
        dataPath = path,
        timeframe = tf,
        indicators = indicators,
        trainTestSplit = 0.8,
        seed = randomSeed,
        train = True
)
valid_dataset = CryptoDataset(
        dataPath = path,
        timeframe = tf,
        indicators = indicators,
        trainTestSplit = 0.8,
        seed = randomSeed,
        train = False
)
