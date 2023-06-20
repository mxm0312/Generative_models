import torch
import numpy as np
import torch.utils
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

""" show error figure """
def plot_train(train_loss):
    fig, axes = plt.subplots(1, 1, figsize=(5, 3))

    axes.set_title('Loss')
    axes.plot(train_loss, label='train')
    axes.legend()

""" Show latent space """
def plot_latent(loader, model, classes=True):
    x = []
    y = []
    targets = []
    for X_batch, Y_batch in loader:
        z = model.encoder(X_batch)
        z = z.detach().numpy()
        x.append(z[0][0])
        y.append(z[0][1])
        targets.append(Y_batch[0])
    if classes:
        plt.scatter(x, y, c=targets, cmap='tab10')
        plt.colorbar()
    else:
        plt.scatter(x, y)

def plot_train(train_loss, val_loss, val_accuracy):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].set_title('Loss')
    axes[0].plot(train_loss, label='train')
    axes[0].plot(val_loss, label='validation')
    axes[0].legend()

    axes[1].set_title('validation accuracy')
    axes[1].plot(val_accuracy)

def plot_train(train_loss):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.set_title('Loss')
    axes.plot(train_loss, label='train')
