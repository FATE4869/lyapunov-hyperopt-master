import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataloader import MNIST_dataloader
from model import LSTM, GRU
from tl_lyapunov import calc_LEs_an
import pickle
import copy
import random

model_type = 'lstm'
hidden_size = 32
trials_num = 2
trials = pickle.load(open(f'trials/{model_type}/models/{model_type}_{hidden_size}_trials_{trials_num}.pickle','rb'))

print(len(trials))
