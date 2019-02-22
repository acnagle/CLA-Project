from sklearn import preprocessing
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
import numpy as np
import errno
import os
import sys
import copy


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
        self.fc6 = nn.Linear(hidden_size, output_size)  # previously, this was output_size
        # self.relu6 = nn.ReLU()  # previously, this was the line which was commented out
        # self.fc7 = nn.Linear(hidden_size, hidden_size)
        # self.relu7 = nn.ReLU()
        # self.fc8 = nn.Linear(hidden_size, hidden_size)
        # self.relu8 = nn.ReLU()
        # self.fc9 = nn.Linear(hidden_size, output_size)
        #         self.relu9 = nn.ReLU()
        #         self.fc10 = nn.Linear(hidden_size, hidden_size)
        #         self.relu10 = nn.ReLU()
        #         self.fc11 = nn.Linear(hidden_size, hidden_size)
        #         self.relu11 = nn.ReLU()
        #         self.fc12 = nn.Linear(hidden_size, output_size)
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
        # out = self.relu6(out)
        # out = self.fc7(out)
        # out = self.relu7(out)
        # out = self.fc8(out)
        # out = self.relu8(out)
        # out = self.fc9(out)
        #         out = self.relu9(out)
        #         out = self.fc10(out)
        #         out = self.relu10(out)
        #         out = self.fc11(out)
        #         out = self.relu11(out)
        #         out = self.fc12(out)
        out = self.sig1(out)
        return out


def main():
    # data processing
    sample_bias = 0  # adjust the difference in the number of the two types of samples (no algae vs algae)
    test_size = 0.2
    batch_size = 100  # batch size for the DataLoaders. previously was 100

    # NN model
    num_features = 17
    input_size = num_features  # size of input layer
    multiplier = 100  # multiplied by num_features to determine the size of each hidden layer. previously was 100
    hidden_size = multiplier * input_size
    output_size = 1
    learning_rate = 0.01  # learning rate of optimizer. previously was 0.01
    num_epochs = 100  # number of epochs

    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    # define data and destination paths
    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/neural-network/"
    data_path = "/Users/Alliot/Documents/CLA-Project/Data/data-sets/"
    data_set = "data_2017_summer"

    # if dest_path does not exist, create it
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # load data sets
    X = np.load(data_path + data_set + ".npy")
    y = np.load(data_path + data_set + "_labels.npy")

    # manipulate data set. labels are converted to -1, +1 for binary classification; samples are removed uniformly
    # from the data set so that the disproportionately large number of negative samples (no algae) does
    # not bias the model.

    num_alg = 0  # count the number of algae instances
    num_no_alg = 0  # count the number of no algae instances

    # Convert labels to binary: -1 for no algae and 1 for algae
    for i in range(0, len(y)):
        if y[i] == 0:
            num_no_alg += 1
        if y[i] == 1 or y[i] == 2:
            y[i] = 1
            num_alg += 1

    # oversample the data set by randomly adding occurences of algae until the difference between the number of algae
    # samples and no algae samples equals sample_bias (defined below)
    idx = 0
    sample_bias = 0
    length_y = len(y)
    while num_alg != (num_no_alg + sample_bias):
        # circle through the data sets until the difference of num_no_alg and num_alg equals
        # the value specified by sample_bias
        if idx == (length_y - 1):
            idx = 0

        if y[idx] == 1:
            if np.random.rand() >= 0.5:  # add this sample with some probability
                y = np.append(y, y[idx])
                X = np.append(X, np.reshape(X[idx, :], newshape=(1, num_features)), axis=0)
                num_alg += 1
            else:
                idx += 1
        else:
            idx += 1

    # standardize data: remove the mean and variance in each sample
    num_splits = 2  # do not change
    sss = model_selection.StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size)

    idx, _ = sss.split(X, y);
    train_idx = idx[0]
    test_idx = idx[1]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train = preprocessing.scale(X_train, axis=1, with_mean=True, with_std=True)
    X_test = preprocessing.scale(X_test, axis=1, with_mean=True, with_std=True)

    # convert numpy arrays to pytorch tensors
    train_set_size = X_train.shape
    test_set_size = X_test.shape
    X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    # convert pytorch tensors to pytorch TensorDataset
    train_set = utils.TensorDataset(X_train, y_train)
    test_set = utils.TensorDataset(X_test, y_test)

    # create DataLoaders
    train_loader = utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = utils.DataLoader(test_set, batch_size=test_set_size[0], shuffle=True)

    model = CLANet(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=1, dampening=0)
    model.double()  # cast model parameters to double

    model.train()  # training mode
    training_loss = []
    avg_error = 0
    avg_error_vec = []
    best_avg_error = 1

    for epoch in range(num_epochs):
        print("Epoch: %d/%d" % (epoch + 1, num_epochs))

        for i, (samples, labels) in enumerate(train_loader):
            samples = Variable(samples)
            labels = Variable(labels)
            output = model(samples)  # forward pass
            output = torch.flatten(output)  # resize predicted labels
            labels = labels.type(torch.DoubleTensor)

            loss = criterion(output, labels)  # calculate loss
            optimizer.zero_grad()  # clear gradient
            loss.backward()  # calculate gradients
            optimizer.step()  # update weights

            # calculate and print error
            out = output

            for j in range(0, out.size()[0]):
                if out[j] < 0.5:
                    out[j] = 0
                else:
                    out[j] = 1
            error = 1 - torch.sum(output == labels).item() / labels.size()[0]
            avg_error += error
            training_loss.append(loss.data.numpy())
            print("  Iteration: %d/%d, Loss: %g, Error: %0.4f" %
                  (i + 1, np.ceil(X_train.size()[0] / batch_size).astype(int), loss.item(), error))

        avg_error = avg_error / np.ceil(X_train.size()[0] / batch_size)
        avg_error_vec.append(avg_error)
        print("Average Error for this Epoch: %0.4f" % avg_error)

        if avg_error < best_avg_error:
            print("found a better model!")
            best_avg_error = avg_error
            best_model = copy.deepcopy(model)

        avg_error = 0

        best_model.eval()

        for i, (samples, labels) in enumerate(test_loader):
            samples = Variable(samples)
            labels = Variable(labels)
            predictions = best_model(samples)
            predictions = torch.flatten(predictions)
            labels = labels.type(torch.DoubleTensor)

            for j in range(0, predictions.size()[0]):
                if predictions[j] < 0.5:
                    predictions[j] = 0
                else:
                    predictions[j] = 1

            error = 1 - torch.sum(predictions == labels).item() / labels.size()[0]

            print("Testing set Error: %0.4f" % error)

        model_path = "./torch_model_2_22_19_lr=" + str(learning_rate) + ",batch_size=" + str(batch_size) + \
                     ",multiplier=" + str(multiplier) + "num_epochs=" + str(num_epochs) + ",process=" + sys.argv[1] + \
                     "_dict.pt"

        torch.save(model.state_dict(), model_path)


if __name__ == "__main__": main()
