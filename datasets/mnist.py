from torch.utils.data import Dataset, DataLoader, Subset
import tensorflow as tf
import torch
import numpy as np

import cv2

class MNISTDataset(Dataset):

    def __init__(self, size: int, rescale=False, transform=None):

        self.transform = transform
        self.rescale = rescale

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()

        self.data = x_train[:size]
        self.target = y_train[:size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.data[idx]
        target = self.target[idx]

        if self.rescale:
            image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            image = image.reshape(1, image.shape[0], image.shape[1])
        else:
            image = image.reshape(image.shape[0] * image.shape[1])

        image = torch.tensor(image, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return image, target
