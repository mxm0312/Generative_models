import torch
import numpy as np
import torch.utils
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

def subset_ind(dataset, ratio):
  return np.random.choice(len(dataset), size=int(ratio*len(dataset)), replace=False)
