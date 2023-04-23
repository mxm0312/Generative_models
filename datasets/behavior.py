import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

PCA_COMPONENTS = 105

class BehaviourDataset(Dataset):
    """Behaviour dataset."""

    def __init__(self, csv_file, harmful_only=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file behaviour
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        # read csv
        df = pd.read_csv(csv_file)
        df = df.drop(['Unnamed: 0'], axis=1)
        if harmful_only:
            df = df[df['tag'] == 1]
        data = df.to_numpy(dtype='float')

        self.target = data[:, -1]
        data = data[:, :-1]

        # preproccess
        scaler = MinMaxScaler()
        pca = PCA(n_components=PCA_COMPONENTS)

        data = scaler.fit_transform(data[:, :-1])
        self.data = pca.fit_transform(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        vect = self.data[idx]
        target = self.target[idx]


        vect = torch.tensor(vect, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return vect, target
