from torch.utils.data import Dataset
import pandas as pd


class ClassifierDataSet(Dataset):
    def __init__(self, file_name):
        data = pd.read_csv(file_name)
        self.x = data['utt']
        self.y = data['index']

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class OODClassifierDataSet(Dataset):
    def __init__(self, file_name):
        data = pd.read_csv(file_name)
        self.x = data['utt']
        self.y = data['index']
        
        if data.get('weight', None) is not None:
            self.weight = data['weight']
        else:
            self.weight = [1] * len(self.x)


    def __getitem__(self, index):
        return self.x[index], self.y[index], self.weight[index]

    def __len__(self):
        return len(self.x)


