import numpy as np
from matplotlib import pyplot as plt

from main.CSV_loader import CSV_loader, index_to_list
from main.network.NeuralNetwork import NeuralNetwork

# Funciones auxiliares para calcular el error y accuracy del conjunto de test
def calculateError(realOutputs, expectedOutputs):
    errors = []
    for realOut,expOut in zip(realOutput, expectedOutput):
        error = sum(map(lambda x, y: (x - y)**2, realOut, expOut))
        errors += [error]
    return sum(errors)/len(errors)

def calculateAccuracy(realOutputs, expectedOutputs):
    acc_list = []
    for realOut, expOut in zip(realOutput, expectedOutput):
        val = 0
        if np.argmax(realOut) == np.argmax(expOut):
            val = 1
        acc_list += [val]
    return sum(acc_list)/len(acc_list)

# se carga el dataset de las semillas
dataset_dir = "data/seeds_dataset.csv"
csv_df = CSV_loader(dataset_dir, 7, index_to_list)

# se crea la red para el dataset

seed_nn = NeuralNetwork(learningRate=0.5)
#red neuronal con 1 capa de escondida con 7 neuronas y una capa de salida de 3 neuronas
seed_nn.factory(7, [3])


# listas para graficar el error y accuracy de la red
error_list = []
accuracy_list = []

# cada iteracion entrena el conjunto de train 10 veces y luego se evalua el conjunto de test
# y se calcula error y accuracy
for _ in range(10):
    for i in range(10):
        for input, expOut in zip(csv_df.train.inputs, csv_df.train.expected_outputs):
            seed_nn.train(input, expOut)

    realOutput = []
    expectedOutput = []
    for input, expOut in zip(csv_df.test.inputs, csv_df.test.expected_outputs):
        expectedOutput += [expOut]
        realOutput += [seed_nn.forwardFeed(input)]

    error = calculateError(realOutput, expectedOutput)
    accuracy = calculateAccuracy(realOutput, expectedOutput)

    error_list += [error]
    accuracy_list += [accuracy]


# se grafican el error y el accuracy vs el numero de epochs
epoch_list = list(range(10,110,10))
plt.plot(epoch_list, error_list)
plt.xlabel("Número de epochs")
plt.ylabel("Error cuadrático")
plt.title("Error cuadrático por número de epochs con 1 capa escondida y lr=0.5")
plt.show()
plt.plot(epoch_list, accuracy_list)
plt.xlabel("Número de epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy por número de epochs con 1 capa escondida y lr=0.5")
plt.show()
