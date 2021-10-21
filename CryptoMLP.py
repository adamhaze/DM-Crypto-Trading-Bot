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
