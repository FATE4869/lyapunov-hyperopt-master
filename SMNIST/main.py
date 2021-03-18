import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataloader import MNIST_dataloader
from model import RNN
from tl_lyapunov import calc_LEs_an
import pickle
def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    sequence_length = 28
    input_size = 28

    num_layers = 1
    num_classes = 10
    batch_size = 100
    num_epochs = 10
    learning_rate = 0.01

    num_trials = 100

    a_s = [0.1, 0.5, 1, 2, 5]
    for a in a_s:
        trials = {}
        for num_trial in range(num_trials):
            print("a: ", a, "num_trial: ", num_trial)
            hidden_size = 32
            trial = {}
            model = RNN(input_size, hidden_size, num_layers, num_classes, a, device).to(device)
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            train_dataloader = MNIST_dataloader(batch_size, train=True)
            test_dataloader = MNIST_dataloader(batch_size, train=False)
            # Train the model
            total_step = len(train_dataloader.dataloader)

            for epoch in range(num_epochs):
                model.train()
                for i, (images, labels) in enumerate(train_dataloader.dataloader):
                    images = images.reshape(-1, sequence_length, input_size).to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs, hts = model(images)
                    loss = criterion(outputs, labels)
                    # print(LEs, rvals)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # if (i + 1) % 300 == 0:
                    #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for i, (images, labels) in enumerate(test_dataloader.dataloader):
                        images = images.reshape(-1, sequence_length, input_size).to(device)
                        labels = labels.to(device)
                        outputs, _ = model(images)

                        h = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
                        c = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
                        params = (images, (h, c))
                        if i == 0:
                            LEs, rvals = calc_LEs_an(*params, model=model)

                        loss = criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    print('Epoch [{}/{}] Loss: {}, Test Accuracy: {} %'.format(epoch + 1, num_epochs, loss, 100 * correct / total))
                trial[epoch] = {"LEs": LEs, "val_loss": loss}
            trials[num_trial] = trial
        pickle.dump(trials, open('trials/lstm_{}_uni_{}.pickle'.format(hidden_size, a), 'wb'))


if __name__ == "__main__":
    main()



