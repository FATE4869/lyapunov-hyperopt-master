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

def plotting_target(tl_learner):
    val_loss = tl_learner.testing_stats[training_epochs - 1]['val_loss']
    # Low Error
    # if val_loss < 0.05:

    # High Error
    if val_loss > 0.5:
        # Mid Error
        # if val_loss> 0.15 and val_loss < 0.25:
        plt.figure()
        start_point = 1200 * 7
        end_point = 1200 * 9
        xrange = range(tl_learner.testing_stats[training_epochs - 1]['inputs'].size)

        plt.plot(xrange[start_point:end_point], tl_learner.testing_stats[14]['inputs'][0, start_point:end_point],
                 'g', linewidth=3)

        plt.plot(xrange[start_point:end_point], tl_learner.dataloader.train_dataset[start_point:end_point],
                 'r', linewidth=3)

        plt.title("val loss: {}, seed: {}".format(val_loss, seed))
        plt.show()
        print("this")



def main():
    function_type = 'random_4sine'
    isRFORCE = False
    if isRFORCE:
        distribution = 'RFORCE'
    else:
        distribution = "FORCE"

    for trial in range(2, 5):
        trials = {}
        for idx in range(2, 11):
            a = time.perf_counter()
            g = int((1.0 + 0.1 * idx) * 10)/ 10
            for seed in range(0, 20):
                print('g = {}, trial = {}; seed={}'.format(g, trial, seed))
                # npr.seed(seed=seed)

                training_epochs = 15
                testing_epochs = 15
                dt = 0.1
                N = 512
                feed_seq = 200
                train = True
                tl_dataloader = targetLearningDataloader(function_type=function_type, training_epochs=training_epochs,
                                                         testing_epochs=testing_epochs, dt=dt)
                tl_learner = Learner(dataloader=tl_dataloader, N=N, g=g, train=train, isRFORCE=isRFORCE)

                tl_learner.learn()

                LEs_stats = {}
                for i in range(0, training_epochs):

                    LEs_stats[i] = ly.LEs(epochs=i, feed_seq=feed_seq, is_test=True, tl_learner = tl_learner)

                trials[seed] = {"seed": seed, "LEs_stats": LEs_stats, "wo": tl_learner.wo_recording[:, -1]}
                if not os.path.exists('../trials/{}/{}/N_{}/g_{}'.format(distribution, function_type, N, g)):
                    os.makedirs('../trials/{}/{}/N_{}/g_{}'.format(distribution, function_type, N, g))
                pickle.dump(trials, open('../trials/{}/{}/N_{}/g_{}/{}_learner_N_{}_g_{:0.1f}_trial_{}.p'.format
                                 (distribution, function_type, N, g, function_type, N, g, trial), 'wb'))

            b = time.perf_counter()
            print("elapse time: ", b - a)
if __name__ == "__main__":
    main()