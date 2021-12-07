# DM-Crypto-Trading-Bot

## Source Code

``Dataset.py``
This file is where our PyTorch dataset is defined as `class CryptoDataset`. This class stores all our datasets of varying timeframes, along with the labeled 5 minute interval data that we are trying to make predictions on. Here we define the \_\_getitem\_\_ method which tells the PyTorch data loader which lag data from multiple timeframes we want to include with the current data point as a sequence that will pass directly into our Recurrent Neural Network.

``indicator_funcs.py``
This function contains the indicators functions we used to better predict buy, sell, and hold signals. The functions as labeled are RSI (which calculates the RSI factor), MA (which calculates the Moving Average), and the MACD (which calculates the Moving Average Convergence Divergence). All of these indicators are based off past data trends, many of which take in data points from 10-30 periods of data. This indicators have helped stock market and crypto traders make more accurate predictions on what to do with their stock/crypto. 

``models.py``
Here we define `class CryptoRNN` as our model for this project. This model leverages the default PyTorch RNN, but we developed an architecture that best suits our problem space. The most important part of this class is the `forward` method which defines specifically how we execute a forward pass through our RNN.

``train.py``

``test.py``

``websocket_test.py``
