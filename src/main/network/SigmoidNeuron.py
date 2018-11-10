import numpy as np

from main.network.AbstractNeuron import AbstractNeuron

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class SigmoidNeuron(AbstractNeuron):

    def __init__(self, weights, bias):
        super().__init__(weights, bias)
        self.weights = weights
        self.bias = bias
        self.activation_function = sigmoid