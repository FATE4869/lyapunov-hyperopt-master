import numpy as np
import numpy.random as npr
from numpy import linalg as LA
import matplotlib.pyplot as plt


class targetLearningDataloader():
    def __init__(self, function_type, idling_epoch = 1, training_epochs = 15,
                 testing_epochs = 10, dt = 0.1):
        self.function_type = function_type
        self.idling_epoch = idling_epoch
        self.training_epochs = training_epochs
        self.testing_epochs = testing_epochs
        self.dt = dt
        self.signal_time = 120
        self.signal_length = int(self.signal_time / self.dt)
        self.setup()
    def setup(self):
        amp = 1
        freq = 1 / 60

        idling_time = self.signal_time * self.idling_epoch
        training_time = self.signal_time * self.training_epochs
        testing_time = self.signal_time * self.testing_epochs

        simtime0 = np.arange(0, idling_time, self.dt)
        simtime1 = np.arange(0, training_time, self.dt)
        simtime2 = np.arange(0, testing_time, self.dt)

        if "4sine" in self.function_type:
            # generationg 4-sine target functions
            if "random" not in self.function_type:
                # fixed scales and freq
                scale_vec = [1.0, 2.0, 6.0, 3.0]
                freq_vec = [1.0, 2.0, 3.0, 4.0]
            else:
                # random scales and freq
                scale_vec = np.random.randint(0, 10, 4) + 1
                freq_vec = np.random.randint(0, 5, 4) + 1
            ft = (amp / scale_vec[0]) * np.sin(freq_vec[0] * np.pi * freq * simtime1) + \
                 (amp / scale_vec[1]) * np.sin(freq_vec[1] * np.pi * freq * simtime1) + \
                 (amp / scale_vec[2]) * np.sin(freq_vec[2] * np.pi * freq * simtime1) + \
                 (amp / scale_vec[3]) * np.sin(freq_vec[3] * np.pi * freq * simtime1)
            ratio = (np.max(ft) - np.min(ft)) / 2
            transition = (np.max(ft) + np.min(ft)) / 2
            ft = ft / (1 * ratio) - transition

            ft2 = (amp / scale_vec[0]) * np.sin(freq_vec[0] * np.pi * freq * simtime2) + \
                 (amp / scale_vec[1]) * np.sin(freq_vec[1] * np.pi * freq * simtime2) + \
                 (amp / scale_vec[2]) * np.sin(freq_vec[2] * np.pi * freq * simtime2) + \
                 (amp / scale_vec[3]) * np.sin(freq_vec[3] * np.pi * freq * simtime2)
            ft2 = ft2 / (1 * ratio) - transition
            # plt.figure()
            # plt.plot(ft)
            # plt.show()
        elif self.function_type == "2sine":
            # generationg 2-sine target functions

            ft = (amp / 1.0) * np.sin(1.0 * np.pi * freq * simtime1) + \
                 (amp / 2.0) * np.sin(2.0 * np.pi * freq * simtime1)
            scale = np.max(ft)

            ft = ft / (1 * scale)
            ft2 = (amp / 1.0) * np.sin(1.0 * np.pi * freq * simtime2) + \
                  (amp / 2.0) * np.sin(2.0 * np.pi * freq * simtime2)
            ft2 = ft2 / (1 * scale)

        self.idle_simtime = simtime0
        self.train_simtime = simtime1
        self.test_simtime = simtime2
        self.train_dataset = ft
        self.test_dataset = ft2


def main():
    function_type = '4sine'
    epochs = 12
    tl_dataloader = targetLearningDataloader(function_type=function_type)
    # print(tl_dataloader.train_dataset)
    plt.figure()
    plt.scatter(tl_dataloader.train_simtime, tl_dataloader.train_dataset)
    plt.title("Training")
    plt.show()

    plt.figure()
    plt.scatter(tl_dataloader.test_simtime, tl_dataloader.test_dataset)
    plt.title("Testing")
    plt.show()

if __name__ == '__main__':
    main()

# plt.figure()
# [D,V] = LA.eig(M)
# plt.scatter(np.real(D),np.imag(D))
# plt.show()
#



