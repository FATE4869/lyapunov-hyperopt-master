import torch
from torch.autograd import Variable
from torch.nn import RNN, GRU, LSTM
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
import gc
import math
import pickle

def LE_stats(LE, save_file=False, file_name='LE.p'):
    mean, std = (torch.mean(LE, dim=0), torch.std(LE, dim=0))
    if save_file:
        pkl.dump((mean, std), open(file_name, "wb"))
    return mean, std


def plot_spectrum(LE, model_name, k_LE=100000, plot_size=(10, 7), legend=[], show=False):
    k_LE = max(min(LE.shape[1], k_LE), 1)
    LE_mean, LE_std = LE_stats(LE)
    f = plt.figure(figsize=plot_size)
    x = range(1, k_LE + 1)
    plt.title('Mean LE Spectrum for ' + model_name)
    f = plt.errorbar(x, LE_mean[:k_LE].to(torch.device('cpu')), yerr=LE_std[:k_LE].to(torch.device('cpu')), marker='.',
                     linestyle=' ', markersize=7, elinewidth=2)
    plt.xlabel('Exponent #')
    if show:
        plt.show()

def main():
    a = 5
    hidden_size =32
    model_name = 'lstm_{}_uni_{}'.format(hidden_size, a)
    trials = pickle.load(open('trials/' + model_name + '.pickle', 'rb'))
    print(len(trials))
    plot_spectrum(trials[2][1]['LEs'], model_name, show=True)


if __name__ == '__main__':
    main()