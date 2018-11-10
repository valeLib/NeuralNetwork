

class Dataset_input:

    def __init__(self, dataframe, class_column, function):
        self.df = dataframe

        # Shuffle the dataset
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        # get the column of classes, and map the respective function
        self.expected_outputs = self.df.iloc[:, [class_column]].values.tolist()
        self.expected_outputs = list(map(function, self.expected_outputs))

        # Normalize dataset
        self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())

        # Save dataframe as list
        columns = [x for x in range(len(self.df.columns)) if x != class_column]
        self.inputs = self.df.iloc[:, columns].values.tolist()

    def format_expected_outputs(self, function):
        return list(map(function, self.expected_outputs))


