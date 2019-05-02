from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import roc_curve, precision_recall_curve
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
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
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.tanh4 = nn.Tanh()
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_size, output_size)  # previously, this was output_size
        #         self.tanh6 = nn.Tanh()                             # previously, this was the line which was commented out
        #         self.fc7 = nn.Linear(hidden_size, output_size)
        #         self.relu7 = nn.ReLU()
        #         self.fc8 = nn.Linear(hidden_size, hidden_size)
        #         self.relu8 = nn.ReLU()
        #         self.fc9 = nn.Linear(hidden_size, output_size)
        #         self.relu9 = nn.ReLU()
        #         self.fc10 = nn.Linear(hidden_size, hidden_size)
        #         self.relu10 = nn.ReLU()
        #         self.fc11 = nn.Linear(hidden_size, hidden_size)
        #         self.relu11 = nn.ReLU()
        #         self.fc12 = nn.Linear(hidden_size, output_size)
        self.sig1 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.tanh4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.fc6(out)
        #         out = self.tanh6(out)
        #         out = self.fc7(out)
        #         out = self.relu7(out)
        #         out = self.fc8(out)
        #         out = self.relu8(out)
        #         out = self.fc9(out)
        #         out = self.relu9(out)
        #         out = self.fc10(out)
        #         out = self.relu10(out)
        #         out = self.fc11(out)
        #         out = self.relu11(out)
        #         out = self.fc12(out)
        out = self.sig1(out)
        return out


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    # define data and destination paths
    data_path_hourly = 'data/'
    X_2015 = np.load(data_path_hourly + 'hourly_X_2015.npy')
    X_2016 = np.load(data_path_hourly + 'hourly_X_2016.npy')
    X_2017 = np.load(data_path_hourly + 'hourly_X_2017.npy')
    X_2018 = np.load(data_path_hourly + 'hourly_X_2018.npy')

    y_2015 = np.load(data_path_hourly + 'hourly_y_2015.npy')
    y_2016 = np.load(data_path_hourly + 'hourly_y_2016.npy')
    y_2017 = np.load(data_path_hourly + 'hourly_y_2017.npy')
    y_2018 = np.load(data_path_hourly + 'hourly_y_2018.npy')

    X = np.vstack((X_2015, X_2016, X_2017, X_2018)).astype(float)
    y = np.hstack((y_2015, y_2016, y_2017, y_2018))
    # X = X_2018.astype(float)
    # y = y_2018

    # labels are converted to 0, +1 for binary classification; samples are removed uniformly
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

    # define hyperparameters
    sample_bias = 0  # adjust the difference in the number of the two types of samples (no algae vs algae)
    test_size = 0.2
    batch_size = 16  # batch size for the DataLoaders
    num_features = X.shape[0]
    input_size = num_features  # size of input layer
    multiplier = 100  # multiplied by num_features to determine the size of each hidden layer
    hidden_size = multiplier * input_size
    output_size = 2
    learning_rate = 0.001  # learning rate of optimizer
    num_epochs = 200  # number of epochs

    # standardize data: remove the mean and variance in each sample
    num_splits = 2  # do not change
    sss = model_selection.StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size)

    idx, _ = sss.split(X, y)
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
    train_set = utils.TensorDataset(X_train.cuda(), y_train.cuda())
    test_set = utils.TensorDataset(X_test.cuda(), y_test.cuda())

    # create DataLoaders
    train_loader = utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = utils.DataLoader(test_set, batch_size=test_set_size[0], shuffle=True)

    # determine weight vector for the each class; used in the loss function
    num_pos = y.tolist().count(0)
    weight_pos = 10
    num_neg = y.tolist().count(1)
    weight_neg = 1
    weight = torch.tensor([weight_neg, weight_pos]).type(torch.DoubleTensor)

    # define model
    model = CLANet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=1, dampening=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    model.double();  # cast model parameters to double

    model.train()  # training mode
    model.cuda()
    avg_error = 0
    best_avg_error = 1

    avg_error_vec = []

    for epoch in range(num_epochs):
        print("Epoch: %d/%d" % (epoch + 1, num_epochs))

        for i, (samples, labels) in enumerate(train_loader):
            samples = Variable(samples)
            labels = Variable(labels)
            output = model(samples.cuda())  # forward pass
            #         output = torch.flatten(output)         # resize predicted labels
            labels = labels.type(torch.long)

            loss = criterion(output, labels)  # calculate loss
            optimizer.zero_grad()  # clear gradient
            loss.backward()  # calculate gradients
            optimizer.step()  # update weights

            # calculate and print error
            out = output

            out = torch.argmax(output, dim=1)  # convert output of network to labels for accuracy calculation
            error = 1 - (torch.sum(out == labels).item() / labels.size()[0])
            avg_error += error

            print("  Batch: %d/%d, Loss: %g, Error: %0.4f" %
                  (i + 1, np.ceil(X_train.size()[0] / batch_size).astype(int), loss.item(), error))

        avg_error = avg_error / np.ceil(X_train.size()[0] / batch_size)
        avg_error_vec.append(avg_error)
        print("Average Error for this Epoch: %0.4f" % avg_error)

        if avg_error < best_avg_error:
            print("found a better model!")
            best_avg_error = avg_error
            best_model = copy.deepcopy(model)

        avg_error = 0

    # evaluate model on test data
    best_model.eval()
    best_model.cuda()
    conf = torch.tensor([]).cuda()
    target = torch.tensor([]).cuda()

    for i, (samples, labels) in enumerate(test_loader):
        samples = Variable(samples)
        labels = Variable(labels)
        predictions = best_model(samples.cuda())
        #     predictions = torch.flatten(predictions)
        labels = labels.type(torch.long)

        predictions = torch.argmax(predictions, dim=1)  # convert output of network to labels for accuracy calculation

        error = 1 - (torch.sum(predictions == labels).item() / labels.size()[0])

        print("Testing set Error: %0.4f" % error)

    # convert to numpy arrays
    conf = conf.detach().numpy()
    labels = labels.numpy()

    # sort arrays according to the predicted confidence (high confidence to low confidence)
    sort_idx = np.argsort(-conf, kind='mergesort')
    conf = conf[sort_idx]
    labels = labels[sort_idx]

    # model_path = dest_path + "torch_model_4_4_19_lr=" + str(learning_rate) + "_hourly_dict.pt"

    # plot roc curve
    fpr, tpr, _ = roc_curve(labels, conf, pos_label=1)

    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr)
    plt.xlabel('False Postive Rate (FPR)', fontsize=20)
    plt.ylabel('True Positive Rate (TPR)', fontsize=20)
    plt.axis([0, 1, 0, 1.001])
    plt.title('ROC Curve', fontsize=20)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('roc.png')

    # plot pr curve
    precision, recall, _ = precision_recall_curve(labels, conf, pos_label=1)

    plt.figure(figsize=(12, 8))
    plt.plot(recall, precision)
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.axis([0, 1.001, 0, 1])
    plt.title('PR Curve', fontsize=20)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('pr.png')


if __name__ == "__main__":
    main()
