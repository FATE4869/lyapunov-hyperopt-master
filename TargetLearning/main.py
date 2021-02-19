from dataloader import targetLearningDataloader
import numpy.random as npr
import matplotlib.pyplot as plt
import os
from learner import Learner
import pickle
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.linalg as la
from plotting import plotting
import tl_lyapunov as ly
import time

def main():
    function_type = '4sine'
    for trial in range(0, 10):
        trials = {}
        for idx in range(0, 1):
            a = time.perf_counter()
            g = 1.4 + 0.1 * idx
            for seed in range(0, 50):
                print('trial = {}; seed={}'.format(trial, seed))
                npr.seed(seed=seed)

                training_epochs = 15
                testing_epochs = 10
                dt = 0.1
                N = 512
                feed_seq = 200
                train = True
                tl_dataloader = targetLearningDataloader(function_type=function_type, training_epochs=training_epochs,
                                                         testing_epochs=testing_epochs, dt=dt)

                tl_learner = Learner(dataloader=tl_dataloader, N=N, g=g, train=train)

                tl_learner.learn()
                LEs_stats = {}
                for i in range(0, training_epochs):

                    LEs_stats[i] = ly.LEs(epochs=i, feed_seq=feed_seq, is_test=True, tl_learner = tl_learner)

                trials[seed] = {"seed": seed, "LEs_stats": LEs_stats}
                if not os.path.exists('../RFORCE/{}/N_{}/'.format(function_type, N)):
                    os.makedirs('../RFORCE/{}/N_{}/'.format(function_type, N))
                pickle.dump(trials, open('../RFORCE/{}/N_{}/{}_learner_N_{}_g_{:0.1f}_trial_{}.p'.format
                                 (function_type, N, function_type, N, g, trial), 'wb'))

            b = time.perf_counter()
            print("elapse time: ", b - a)
if __name__ == "__main__":
    main()