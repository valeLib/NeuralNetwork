import unittest

from main.network.NeuralNetwork import NeuralNetwork
from main.network.NeuronLayer import NeuronLayer
from main.network.SigmoidNeuron import SigmoidNeuron
from main.network.aux_methods import truncate


class NNTest(unittest.TestCase):
    def test(self):
        neuron1 = SigmoidNeuron([0.4, 0.3], [0.5])
        neuron2 = SigmoidNeuron([0.3], [0.4])

        layer1 = NeuronLayer([neuron1])
        layer2 = NeuronLayer([neuron2])
        layer1.nextLayer = layer2
        layer2.previousLayer = layer1

        nn = NeuralNetwork(neuronLayers=[layer1, layer2])

        nn.train([1, 1], [1])

        assert truncate(neuron1.bias[0],3) == 0.502, "Fallo bias en neurona 1"
        assert truncate(neuron1.weights[0], 3) == 0.402, "Fallo peso 1 en neurona 1"
        assert truncate(neuron1.weights[1], 3) == 0.302, "Fallo peso 2 en neurona 1"

        assert truncate(neuron2.bias[0], 3) == 0.439, "Fallo bias en neurona 2"
        assert truncate(neuron2.weights[0], 3) == 0.330, "Fallo peso 1 en neurona 2"

if __name__ == '__main__':
    unittest.main()
