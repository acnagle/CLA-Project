import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import model_selection
import numpy as np
import sys


class CLANet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CLANet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.sig1 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.fc6(out)
        out = self.sig1(out)
        return out


def main():
    # hyperparameters
    # the entire learning rate "vector" consists of 112 evenly spaced values between 0.0001 and 0.1 inclusive
    # learning_rate = 0.0001 + sys.argv[1] * 0.0009
    learning_rate = np.linspace(start=0.0001, stop=0.1, endpoint=True, num=100)
    outer_fold = sys.argv[1]    # the fold of the outer CV that the inner CV will run on
    num_epochs = 5              # number of epochs  # todo make epochs = 100 or something that will work with nested cv
    batch_size = 100            # batch size for the DataLoaders. Predetermined. DO NOT CHANGE
    multiplier = 100            # determines size of hidden layers
    num_features = 17
    input_size = num_features   # size of input layers
    hidden_size = multiplier * input_size
    output_size = 1
    num_outer_folds = 10
    num_inner_folds = 9               # number of folds for the inner loop of nested CV (10 folds on the outer)

    # Load data
    X = np.load("X.npy")
    y = np.load("y.npy")
    # TODO BE SURE TO INCLUDE INOFRMATION ABOUT WHAT OUTER LOOP NUMBER IS BEFORE SAVING RESULTS
    # load DataLoaders  # TODO MAKE A TRAIN/VALIDATION/TEST SPLIT
    # train_loader = torch.load("train_loader.pt")
    # test_loader = torch.load("test_loader.pt")

    # train neural network
    training_loss = []
    avg_error = 0
    best_test_error = 1
    # num_training_samples = len(train_loader.dataset)

    # find the outer fold
    skf = model_selection.StratifiedKFold(n_splits=num_outer_folds)
    count = 1
    for train_index_outer, _ in skf.split(X, y):
        if count == outer_fold:
            X_train_outer = X[train_index_outer]
            y_train_outer = y[train_index_outer]
        else:
            count += 1

    # instantiate neural networks
    models = []
    criterion = []
    for i in range(len(learning_rate)):
        models.append(CLANet(input_size, hidden_size, output_size))
        criterion.append(nn.BCELoss())

        optimizer = torch.optim.SGD(models[i].parameters(), lr=learning_rate[i], nesterov=True, momentum=1, dampening=0)
        models[i].double()  # cast model parameters to double

    print("~~~~~~~~~~~~~~~~~~~~~~~~~ Performing nested CV to tune learning rate ~~~~~~~~~~~~~~~~~~~~~~~~~")

    for r in range(len(learning_rate)):
        print("Learning Rate = ", learning_rate[r], "\n")

        for train_index_inner, tune_index in skf.split(X_train_outer, y_train_outer):
            X_train_inner, X_test = X[train_index_outer], X[tune_index]
            y_train_inner, y_test = y[train_index_outer], y[tune_index]

        for epoch in range(num_epochs):
            print("Epoch: %d/%d" % (epoch + 1, num_epochs))

            model.train()
            for i, (samples, labels) in enumerate(train_loader):
                samples = Variable(samples)
                labels = Variable(labels)
                output = model(samples)         # forward pass
                output = torch.flatten(output)  # resize predicted labels
                labels = labels.type(torch.DoubleTensor)

                loss = criterion(output, labels)  # calculate loss
                optimizer.zero_grad()  # clear gradient
                loss.backward()  # calculate gradients
                optimizer.step()  # update weights

                # calculate and print error
                output = torch.round(output)

                error = 1 - torch.sum(output == labels).item() / labels.size()[0]
                avg_error += error
                training_loss.append(loss.data.numpy())
                print("  Iteration: %d/%d, Loss: %g, Error: %0.4f" %
                      (i + 1, np.ceil(num_training_samples / batch_size).astype(int), loss.item(), error))

            print("Average Error for this Epoch: %0.4f" % (avg_error / np.ceil(num_training_samples / batch_size)))
            avg_error = 0

            model.eval()

            for i, (samples, labels) in enumerate(test_loader):
                samples = Variable(samples)
                labels = Variable(labels)
                predictions = model(samples)
                predictions = torch.flatten(predictions)
                labels = labels.type(torch.DoubleTensor)

                predictions = torch.round(predictions)

                error = 1 - torch.sum(predictions == labels).item() / labels.size()[0]

                print("\nTesting set Error: %0.4f\n" % error)

                if error < best_test_error:
                    torch.save(model, "./best_model_on_test_set.pt")
                    best_test_error = error


if __name__ == "__main__": main()
