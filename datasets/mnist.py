from torch.utils.data import Dataset, DataLoader, Subset
import tensorflow as tf
import torch
import numpy as np

class MNISTDataset(Dataset):

    def __init__(self, size: int, transform=None):

        self.transform = transform

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()

        self.data = x_train[:size]
        self.target = y_train[:size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.data[idx]
        target = self.target[idx]

        image = image.reshape(image.shape[0] * image.shape[1])

        image = torch.tensor(image, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return image, target
