import numpy as np
from abc import ABC
import random as rd

class AbstractNeuron(ABC):

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.activation_function = None
        self.delta = None
        self.output = 0
        super().__init__()

    def feed(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation_function(self.output)
        self.output = self.output[0]
        return self.output

    def training(self, desiredOutput, realOutput, learningRate, inputs):
        diff = []
        for x, y in zip(desiredOutput, realOutput):
            diff.append(x - y)

        for i in range(0, len(inputs)):
            self.weights[i] = self.weights[i] + (learningRate * inputs[i] * diff)
        self.bias = self.bias + [map((lambda x: learningRate * x), diff)]

    def adjustBiasUsingLearningRate(self, learningRate):
        self.bias += (learningRate * self.delta)

    def adjustDeltaWith(self, error):
        transferDerivative = self.output * (1.0 - self.output)
        self.delta = error * transferDerivative

    def adjustWeightWithInput(self, inputs, learningRate):
        for i in range(len(inputs)):
            self.weights[i] += learningRate * self.delta * inputs[i]



