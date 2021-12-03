# DM-Crypto-Trading-Bot

## Source Code

``Dataset.py``
This file is where our PyTorch dataset is defined as `class CryptoDataset`. This class stores all our datasets of varying timeframes, along with the labeled 5 minute interval data that we are trying to make predictions on. Here we define the \_\_getitem\_\_ method which tells the PyTorch data loader which lag data from multiple timeframes we want to include with the current data point as a sequence that will pass directly into our Recurrent Neural Network.

``indicator_funcs.py``

``models.py``
Here we define `class CryptoRNN` as our model for this project. This model leverages the default PyTorch RNN, but we developed an architecture that best suits our problem space. The most important part of this class is the `forward` method which defines specifically how we execute a forward pass through our RNN.

``train.py``

``test.py``

``websocket_test.py``
