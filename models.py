import torch
import torch.nn as nn


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

class CryptoRNN(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenDim, nLayers = 1):
        super(CryptoRNN, self).__init__()

        # model parameters
        self.hidden_dim = hiddenDim
        self.n_layers = nLayers
        self.input_size = inputSize
        self.output_size = outputSize

        # Defining the layers
        self.rnn = nn.RNN(self.input_size, self.hidden_dim, self.n_layers, batch_first=True)   
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden