'''
This code will load trained model and calculate the LEs

gru model are stored in 'trials/gru/models/...'
each .pickle file contains 50 trials and each trial is trained with 20 epochs, the model and accuracy
after each epoch of training is saved. You can just load the model to perform the LEs calculation

lstm model:
'''
import pickle
import torch
from dataloader import MNIST_dataloader
import torch.nn as nn
from model import LSTM, GRU
from tl_lyapunov import calc_LEs_an
import copy

device = torch.device('cuda')
def cal_LEs_from_trained_model(N, a=None, trial_num=None):
    model_type = 'lstm'
    if a is None:
        trials = pickle.load(open(f'trials/{model_type}/models/{model_type}_{N}_trials_{trial_num}.pickle','rb'))
    else:
        trials = pickle.load(open(f'trials/{model_type}/models/{model_type}_{N}_uni_{a}.pickle', 'rb'))
    num_trials = len(trials)
    num_epochs = len(trials[0])-1
    batch_size = 100
    sequence_length = 28
    input_size = 28
    test_dataloader = MNIST_dataloader(batch_size, train=False)
    criterion = nn.CrossEntropyLoss()
    num_classes = 10
    num_layers = 1
    LEs_stats = {}

    for i in range(num_trials):
        trial = trials[i][num_epochs - 1]
        model, record_acc, record_loss, init_param = trial['model'], trial['accuracy'],\
                                                     trial['loss'], trial['model'].a
        # Test the model

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for idx, (images, labels) in enumerate(test_dataloader.dataloader):
                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                outputs, _ = model(images)

                h = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
                if model_type == 'lstm':
                    c = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
                    params = (images, (h, c))
                elif model_type == 'gru':
                    params = (images, h)
                if idx == 0:
                    LEs, rvals = calc_LEs_an(*params, model=model, rec_layer=model_type)
                    # print(LEs, rvals)
                    LEs_avg = torch.mean(LEs, dim=0)


                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total_loss += loss * labels.size(0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        LEs_stats[i] = {}
        LEs_stats[i]['LEs'] = LEs_avg
        LEs_stats[i]['accuracy'] = 100 * correct / total
        LEs_stats[i]['loss'] = total_loss / total

        print(f'Loss: {total_loss / total}, Test Accuracy: {100 * correct / total}. a: {init_param},'
              f' record_loss: {record_loss}, record_acc: {record_acc}')
    pickle.dump(LEs_stats, open(f'trials/{model_type}/LEs/{model_type}_{N}_trials_{trial_num}.pickle','wb'))

def main():
    # for gru models
    # N_s = [64]
    # a_s = [2.0, 2.2, 2.5, 2.7, 3.0]
    # for N in N_s:
    #     for a in a_s:
    #         cal_LEs_from_trained_model(N, a)

    # for lstm models
    N_s = [64]
    # a_s = [2.0, 2.2, 2.5, 2.7, 3.0]

    for trial_num in range(2, 7):
        # for a in a_s:
        N = 64
        cal_LEs_from_trained_model(N, trial_num=trial_num)

if __name__ == "__main__":
    main()
