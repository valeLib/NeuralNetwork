import random as rd

from main.network.NeuronLayer import NeuronLayer
from main.network.SigmoidNeuron import SigmoidNeuron


class NeuralNetwork:

    def __init__(self, neuronLayers=None, learningRate=0.5):
        self.neuronLayers = neuronLayers if neuronLayers is not None else []
        self.learningRate = learningRate
        self.outputs = []

    def factory(self, inputSize, layerSizes):
        layerSizes_total = [inputSize] + layerSizes
        layers = []
        for i in range(1,len(layerSizes_total)):
            neurons = []
            for j in range(layerSizes_total[i]):
                weights = [rd.gauss(0,1) for _ in range(layerSizes_total[i-1])]
                neuron = SigmoidNeuron(weights, [0])
                neurons += [neuron]
            layer = NeuronLayer(neurons)
            layers += [layer]
        for i in range(len(layers)-1):
            layers[i].nextLayer = layers[i+1]
            layers[i+1].previousLayer = layers[i]

        self.neuronLayers = layers

    def train(self, inputs, expectedOutputs):
        self.outputs = self.forwardFeed(inputs)
        self.backwardPropagateError(expectedOutputs)
        self.updateWeight(inputs)

    def forwardFeed(self, inputs):
        layerOutput = []
        for neuronLayer in self.neuronLayers:
            if layerOutput == []:
                layerOutput = neuronLayer.feed(inputs)
            else:
                layerOutput = neuronLayer.feed(layerOutput)
        return layerOutput

    def  backwardPropagateError(self, expectedOutputs):
        lastLayer = self.neuronLayers[len(self.neuronLayers) - 1]
        lastLayer.backwardPropagateError(expectedOutputs)

    def updateWeight(self, initialInputs):
        firstHiddenLayer = self.neuronLayers[0]
        firstHiddenLayer.updateWeight(initialInputs, self.learningRate)







