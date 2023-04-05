import torch
import numpy as np
import torch.utils
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

""" Train model function """
def train_model_base(model, train_loader, loss_f, opt, epochs, show_img=False):

    train_loss = []

    for epoch in range(epochs):
        ep_train_loss = []
        start_time = time.time()

        model.train(True) # enable dropout / batch norm
        for X_batch, Y_batch in train_loader:

            predictions = model(X_batch)
            if (show_img):
                clear_output(True)
                plt.imshow(predictions[0].detach().numpy().reshape(28,28))
                plt.show()
            opt.zero_grad()

            loss = loss_f(predictions, X_batch)
            loss.backward()
            opt.step()

            ep_train_loss.append(loss.item())

        print(f'Epoch {epoch + 1}/{epochs}. time: {time.time() - start_time:.3f}s')

        train_loss.append(np.mean(ep_train_loss))

        print(f'train loss: {train_loss[-1]:.6f}')
        print('\n')

    return train_loss

def train_model_vae(model, train_loader, opt, epochs, show_img=False):

    train_loss = []

    for epoch in range(epochs):
        ep_train_loss = []
        start_time = time.time()

        model.train(True) # enable dropout / batch norm
        for X_batch, Y_batch in train_loader:

            predictions = model(X_batch)
            if (show_img):
                clear_output(True)
                plt.imshow(predictions[0].detach().numpy().reshape(28,28))
                plt.show()
            opt.zero_grad()

            loss = ((X_batch - predictions)**2).sum() + model.encoder.kl

            loss.backward()
            opt.step()

            ep_train_loss.append(loss.item())

        print(f'Epoch {epoch + 1}/{epochs}. time: {time.time() - start_time:.3f}s')

        train_loss.append(np.mean(ep_train_loss))

        print(f'train loss: {train_loss[-1]:.6f}')
        print('\n')

    return train_loss

def train_model(model, train_loader, val_loader, loss_f, opt, epochs):


    train_loss = []
    val_loss = []
    val_accuracy = []

    for epoch in range(epochs):

        ep_train_loss = []
        ep_val_loss = []
        ep_val_accuracy = []
        start_time = time.time()

        model.train(True) # enable dropout / batch norm
        for X_batch, Y_batch in train_loader:

            predictions = model(X_batch)
            opt.zero_grad()

            loss = loss_f(predictions, torch.reshape(Y_batch, (X_batch.shape[0], 1)))
            loss.backward()
            opt.step()
            ep_train_loss.append(loss.item())

        model.train(False)
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                predictions = model(X_batch)

                loss = loss_f(predictions, torch.reshape(Y_batch, (X_batch.shape[0], 1)))
                ep_val_loss.append(loss.item())
                ep_val_accuracy.append(np.mean( (torch.round(predictions) == Y_batch).numpy() ))

        print(f'Epoch {epoch + 1}/{epochs}. time: {time.time() - start_time:.3f}s')

        train_loss.append(np.mean(ep_train_loss))
        val_loss.append(np.mean(ep_val_loss))
        val_accuracy.append(np.mean(ep_val_accuracy))

        print(f'train loss: {train_loss[-1]:.6f}')
        print(f'val loss: {val_loss[-1]:.6f}')
        print(f'validation acc: {val_accuracy[-1]:.6f}')
        print('\n')

    return train_loss, val_loss, val_accuracy


""" show error figure """
def plot_train(train_loss):
    fig, axes = plt.subplots(1, 1, figsize=(5, 3))

    axes.set_title('Loss')
    axes.plot(train_loss, label='train')
    axes.legend()

""" Show latent space """
def plot_latent(loader, model):
    x = []
    y = []
    targets = []
    for X_batch, Y_batch in loader:
        z = model.encoder(X_batch)
        z = z.detach().numpy()
        x.append(z[0][0])
        y.append(z[0][1])
        targets.append(Y_batch[0])

    plt.scatter(x, y, c=targets, cmap='tab10')
    plt.colorbar()

def plot_train(train_loss, val_loss, val_accuracy):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].set_title('Loss')
    axes[0].plot(train_loss, label='train')
    axes[0].plot(val_loss, label='validation')
    axes[0].legend()

    axes[1].set_title('validation accuracy')
    axes[1].plot(val_accuracy)

def subset_ind(dataset, ratio):
  return np.random.choice(len(dataset), size=int(ratio*len(dataset)), replace=False)
