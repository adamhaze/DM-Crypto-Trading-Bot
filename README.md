# DM-Crypto-Trading-Bot

## Source Code

``Dataset.py``
This file is where our PyTorch dataset is defined as `class CryptoDataset`. This class stores all our datasets of varying timeframes, along with the labeled 5 minute interval data that we are trying to make predictions on. Here we define the \_\_getitem\_\_ method which tells the PyTorch data loader which lag data from multiple timeframes we want to include with the current data point as a sequence that will pass directly into our Recurrent Neural Network.

``indicator_funcs.py``
This function contains the indicators functions we used to better predict buy, sell, and hold signals. The functions as labeled are RSI (which calculates the RSI factor), MA (which calculates the Moving Average), and the MACD (which calculates the Moving Average Convergence Divergence). All of these indicators are based off past data trends, many of which take in data points from 10-30 periods of data. This indicators have helped stock market and crypto traders make more accurate predictions on what to do with their stock/crypto. 

``models.py``
Here we define `class CryptoRNN` as our model for this project. This model leverages the default PyTorch RNN, but we developed an architecture that best suits our problem space. The most important part of this class is the `forward` method which defines specifically how we execute a forward pass through our RNN.

``train.py``
In this file we set all our model hyperparameters, load all our data into the necessary format, and conduct all model training. We split the data into train, validation, and test sets in order to provide validation loss values throughout training, and ultimately test the final model on a small held out test dataset to obtain an accuracy measure. The best model, found during training, is then saved, along with a plot of the train vs validation loss over all training epochs.

``test.py``
Here we load our best performing model, along with previously unseen data that we use to test our models performance, as well as evaluate the trading strategy based on an initial investment and our model-predicted buy/sell/hold states.

``websocket_test.py``
This is the file we used to begin experimenting with the Kraken websocket API

``src/*.ipynb``
This is an accumulation of files that were used to set up the SQL database, debug any issues with the database, and includes methods necessary to connect to the database from your local machine. These files also include all relevant commands for pushing and pulling data to and from the database.

``LiveData.ipynb``
