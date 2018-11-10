class NeuronLayer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.previousLayer = None
        self.nextLayer = None
        self.neuron_outputs = []

    def feed(self, inputs):
        neuron_outputs = []
        for neuron in self.neurons:
            neuron_outputs.append(neuron.feed(inputs))
        self.neuron_outputs = neuron_outputs
        return neuron_outputs

    def backwardPropagateError(self, expectedOutputs):
        for i,neuron in enumerate(self.neurons):
            error = expectedOutputs[i] - neuron.output
            neuron.adjustDeltaWith(error)
        if self.previousLayer != None:
            self.previousLayer.backwardHiddenPropagateError()

    def backwardHiddenPropagateError(self):
        for j in range(len(self.neurons)):
            error = 0
            for nexNeuron in self.nextLayer.neurons:
                error += nexNeuron.weights[j] * nexNeuron.delta
            self.neurons[j].adjustDeltaWith(error)
        if self.previousLayer != None:
            self.previousLayer.backwardHiddenPropagateError()

    def updateWeight(self, initialInputs, learningRate):
        inputs = []
        if self.previousLayer is None:
            inputs = initialInputs
        else:
            inputs = self.previousLayer.neuron_outputs
        for neuron in self.neurons:
            neuron.adjustWeightWithInput(inputs, learningRate)
            neuron.adjustBiasUsingLearningRate(learningRate)
        if self.nextLayer != None:
            self.nextLayer.updateWeight(initialInputs, learningRate)