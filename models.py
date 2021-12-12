import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.utils.data import DataLoader


class CryptoMLP(nn.Module):

	def __init__(self, inputSize, numClass, hiddenLayers = [3,3]):
		super(CryptoMLP, self).__init__()

		self.input_size = inputSize
		self.output_size = numClass
		self.hidden_size = [int(i * self.input_size) for i in hiddenLayers] # scale input size of each hidden layer

		self.input = nn.Linear(self.input_size, self.hidden_size[0])
		self.lrelu = nn.LeakyReLU()
		self.output = nn.Linear(self.hidden_size[-1], self.output_size)

		# define hidden layers
		self.hidden = nn.ModuleList()
		for i in range(len(hiddenLayers) - 1):
			self.hidden.append(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))


	def forward(self, inputTensor):

		# set x to match input size for first input layer
		x = inputTensor.view(inputTensor.shape[0], -1)

		# Run forward pass
		x = self.input(x)
		x = self.lrelu(x)
		for i in range(len(self.hidden)):
			x = self.hidden[i](x)
			x = self.lrelu(x)
		x = self.output(x)
		return x




"""
The following model is based off code I sourced online:

https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

GitHub: https://github.com/gabrielloye/RNN-walkthrough/blob/master/main.ipynb
"""
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CryptoRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers = 1, num_classes = 3):
        super(CryptoRNN, self).__init__()

        # model parameters
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.n_classes = num_classes
        self.input_size = input_size
        
        # Defining the layers
        # X -> (batch_size, sequence_length, input_size) --- shape of input X
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu', batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.drop = nn.Dropout(p = 0.1)
        # self.soft = nn.Softmax(dim=1)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_size)
        out = out[:, -1, :]
        # print(out)
        out = self.drop(out)
        # out = self.soft(out)
        # print(out)
        out = self.fc(out)
        # print(out)
        
        
        return out
    
    def init_hidden(self, batch_size):
        # initialize tensor of 0's for first hidden state
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return hidden