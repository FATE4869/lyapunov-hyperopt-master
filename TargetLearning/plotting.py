import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
import tl_lyapunov as ly

def plotting(trial, g, epochs, feed_seq, function_type, N,  test, repeat, load_model, trained_model=True, tl_learner = None):
    LEs_recording = np.zeros([repeat, N])
    if load_model:
        if trained_model:
            tl_learner = pickle.load(open('../Models/Target_Learning/{}_learner_N_{}_g_{}_trial_{}_trained_e_{}.p'.format
                                          (function_type, N, g, trial, epochs), 'rb'))
        else:
            tl_learner = pickle.load(open('../Models/Target_Learning/{}_learner_N_{}_g_{}_trial_{}_untrained_e_{}.p'.format
                                          (function_type, N, g, trial, epochs), 'rb'))

    print(dir(tl_learner))


    plt.figure()
    for i in range(repeat):

        starting_point = tl_learner.dataloader.signal_length * (i)
        if test:
            h0 = tl_learner.testing_stats[epochs]['hidden_states'][:, 0]
            # h0 = tl_learner.h_testing_recording[:,0]
            x_in = tl_learner.testing_stats[epochs]['inputs'][:, :feed_seq]

            # x_in = tl_learner.input_testing_recording[:,:feed_seq]
        else:
            h0 = tl_learner.h_training_recording[:, starting_point]
            x_in = tl_learner.input_training_recording[:, starting_point: starting_point + feed_seq]
        print(x_in)
        h0 = torch.unsqueeze(torch.unsqueeze(torch.tensor(h0), 0), 0)  # h0 = [num hidden layer, batch size, hidden size]
        x_in = torch.unsqueeze(torch.transpose(torch.tensor(x_in), 0, 1), 0)  # x_in = [batch_size, feed_seq, input_size]
        print(h0.size)
        print(x_in.shape)
        LEs, rvals = ly.calc_LEs_an(x_in, h0, learner=tl_learner)
        print(LEs.shape)

        LE_mean, LE_std = ly.LE_stats(LEs)
        stats = {'LEs': LEs, 'LE_mean': LE_mean, 'LE_std': LE_std, 'rvals': rvals}
        if trained_model:
            pickle.dump(stats, open('../LE_stats/{}_LE_stats_N_{}_trial_{}_trained_e_{}.p'.format(function_type, N, trial, epochs), 'wb'))

        LEs = torch.squeeze(LEs, 0).cpu().detach().numpy()
        LEs_recording[i, :] = LEs[:]
        x_axis = np.linspace(0, len(LEs), num = len(LEs), endpoint=False)
        # plt.figure()
        plt.scatter(x_axis, LEs)
        # plt.xlim(-5, 10)
        # plt.ylim(-0 , 1)

        plt.xlim(-5, 205)
        plt.ylim(-5, 1)
        if not trained_model:
            plt.title("Before training")
        elif not test:
            plt.title("Training")
        else:
            plt.title("Testing")
    plt.legend(['e1','e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'])
    plt.show()
    print(LEs_recording)

def largestLEs(val_losses, largest_LEs, threshold=0.1):

    bool_arr_bad = val_losses > threshold
    bool_arr_good = val_losses <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]
    print(len(indices_good), len(indices_bad))
    ax0 = plt.subplot(1, 1, 1)
    ax0.scatter(np.ones_like(val_losses[indices_bad]), largest_LEs[indices_bad], s=10, c='r', label="Bad performace", alpha=0.5)
    ax0.scatter(np.ones_like(val_losses[indices_good]) * 1.1, largest_LEs[indices_good], c='g', s=10, label="Good performace", alpha=0.5)
    ax0.axes.xaxis.set_ticks([1.0, 1.1])
    ax0.axes.xaxis.set_ticklabels([])
    #
    ax0.axes.yaxis.set_ticks([-1, -.5, 0, .5, 1])
    ax0.axes.yaxis.set_ticklabels([])
    # # ax0.axes.xaxis.set_visible(False)
    # # ax0.axes.yaxis.set_visible(False)
    #
    # # g_axis = np.ones_like(val_losses) * g
    # # plt.scatter(g_axis, val_losses)
    plt.xlim([0.95, 2])
    plt.ylim([-1.05, 1.25])
    plt.show()

def main():
    val_losses = None
    largest_LEs = None
    for i in range(16):
        # print(i)
        # g = int((1.0 + 0.1 * i) * 10)/ 10
        g = 1.4
        # print(g)
        trial = i
        seed = 0
        N = 512
        input_epoch = 5
        output_epoch = 14
        function_type = 'random_4sine'
        distribution = "FORCE"
        trials = pickle.load(open('../trials/{}/{}/N_{}/g_{}/{}_learner_N_{}_g_{}_trial_{}.p'.format(
            distribution, function_type,N,g, function_type,N,g,trial), 'rb'))

        # print(len(trials))
        for j in range(len(trials)):
            if val_losses is None:
                val_losses = np.array(trials[j]['LEs_stats'][output_epoch]['val_loss'])
                largest_LEs = np.array(max(trials[j]['LEs_stats'][input_epoch]['LEs']))
            else:
                val_losses = np.append(val_losses, trials[j]['LEs_stats'][output_epoch]['val_loss'])
                largest_LEs = np.append(largest_LEs, max(trials[j]['LEs_stats'][input_epoch]['LEs']))
    # print("val_losses: ", val_losses)
    # print("largest_LEs: ", largest_LEs)
    # plt.scatter(val_losses, largest_LEs)
    # plt.show()
    largestLEs(val_losses, largest_LEs, threshold=0.2)
    #
    # bool_arr_bad = val_losses > threshold
    # bool_arr_good = val_losses <= threshold
    # indices_bad = np.where(bool_arr_bad)[0]
    # indices_good = np.where(bool_arr_good)[0]
    # print(len(indices_good), len(indices_bad))
    # ax0 = plt.subplot(1, 1, 1)
    #
    # ax0.scatter(np.ones_like(val_losses[indices_bad]) * g, val_losses[indices_bad], s=50, c='r', label="Bad performace", alpha=0.5)
    # ax0.scatter(np.ones_like(val_losses[indices_good]) * g, val_losses[indices_good], c='g', s=50, label="Good performace", alpha=0.5)
    # ax0.axes.xaxis.set_ticks([1.4])
    # ax0.axes.xaxis.set_ticklabels([])
    #
    # ax0.axes.yaxis.set_ticks([0, .2, .4, .6, .8, 1])
    # ax0.axes.yaxis.set_ticklabels([])
    # # ax0.axes.xaxis.set_visible(False)
    # # ax0.axes.yaxis.set_visible(False)
    #
    # # g_axis = np.ones_like(val_losses) * g
    # # plt.scatter(g_axis, val_losses)
    # plt.xlim([1.35, 2])
    # plt.ylim([-0.05, 1.05])
    # plt.show()



    # plt.figure()
    # for key in [0, 5, 10, 14]:
    # # for key in trials[0]['LEs_stats'].keys():
    # #     print()
    #     x_axis = range(0, len(trials[seed]['LEs_stats'][key]['LEs']))
    #     plt.scatter(x_axis, trials[seed]['LEs_stats'][key]['LEs'])
    #     # print(trials[0]['LEs_stats'][key][LEs])
    # # print(trials[0]['LEs_stats'][0])
    # # plt.legend(['0', '5', '10', '14'])
    # # plt.xlim(-20, 520)
    # # plt.ylim(-12, 2)
    # # plt.title("g = {}, val_loss = {:0.3f}".format(g, trials[trial]['LEs_stats'][14]['val_loss']))
    # # plt.show()

if __name__ == "__main__":
    main()