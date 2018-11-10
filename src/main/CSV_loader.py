import pandas as pd

from main.Dataset_input import Dataset_input


def index_to_list(index, size=3):
    index = index[0]
    return [0 if i != index - 1 else 1 for i in range(size)]


class CSV_loader:

    def __init__(self, csv_dir, class_column, function=index_to_list):

        # Carga csv como dataframe
        df = pd.read_csv(csv_dir)

        train = df.sample(frac=0.8, random_state=200)
        test = df.drop(train.index)

        self.train = Dataset_input(train, class_column, function)
        self.test = Dataset_input(test, class_column, function)
