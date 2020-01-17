#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

import copy
import random
import sys
from PIL import Image

import resnet

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score

# np.random.seed(0)

run = sys.argv[1]
num_iter = int(sys.argv[2])

# ## Read in Data

data = pd.read_json('./data.json')
labels = data[['label']]
data = data.drop('label', axis='columns')

# ## Add Features

data['log_turbidity'] = np.log(data['turbidity'] + 1)

# ## Feature Correlation

corr = data.corr()

# ### Choose Correlated Features to Remove

corr_thresh = 0.80  # threshold for correlation. for any two variables with correlation > thresh, one is removed

thresh = corr.abs() > corr_thresh

keep = copy.deepcopy(data.columns).to_list()

print('Removed features: ')
# keep features whose correlation with other features is <= corr_thresh
for i in range(0, len(thresh.index)):
    for j in range(i+1, len(thresh.columns)):
        if thresh.iloc[i, j]:
            if thresh.columns[j] in keep:
                print('\t', thresh.columns[j])
                keep.remove(thresh.columns[j])

# ### Split Data

df = data[keep]
# df = df[df.index > '2016']   # only keep data after 2015
# labels = labels.loc[df.index]

class AlgalBloomDataset(data_utils.Dataset):

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (numpy array): numpy array of data samples
            labels (numpy array): numpy array of labels for the data samples
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        target = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, target

train_size = 0.70
batch_size = 16
data_aug = False    # data augmentation
learning_rate = 0.1
num_epochs = 450
weighted = False

if weighted:
    class_weights = torch.Tensor([np.bincount(y_train.astype(int))[0] / len(y_train),
                            np.bincount(y_train.astype(int))[1] / len(y_train)])
else:
    class_weights=None


acc_arr = []
f1_arr = []
tpr_arr = []
fpr_arr = []
conf_matrix_arr = []

for i in range(num_iter):
    print('Iteration', i)

    # zero pad data set. The input format for the resnet must be 5x5, or 6x6, or 7x7, etc.
    pad_df = copy.deepcopy(df)
    for i in range(36-df.shape[1]):
        pad_df[i] = [1 for _ in range(df.shape[0])]

    vals = pad_df.values
    data_reshape = []
    for i in range(pad_df.shape[0]):
        data_reshape.append(vals[i].reshape(6, 6))

    # Stratified split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test,
        y_train_test,
        train_size=train_size,
        shuffle=True,
        stratify=y_train_test
    )

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if data_aug:
        trnsfrm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomAffine(degrees=(0, 360), translate=(1/6, 1/6), scale=(0.99, 1.01)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ])
    else:
        trnsfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ])

    train_set = AlgalBloomDataset(X_train, y_train, trnsfrm)

    test_set = AlgalBloomDataset(X_test, y_test,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ])
    )

    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

    # ## Define Model

    model = resnet.ResNet9().cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = optim.SGD(model.parameters(), lr=learning_rate, nesterov=False, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[75, 150, 225, 300, 375], gamma=0.1)

    # ## Train Model

    for epoch in range(num_epochs):
        model.train()   # train model

        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 15)

        for samples, target in train_loader:
            opt.zero_grad()
            samples, target = samples.cuda(), target.cuda()
            output = model(samples)
            loss = criterion(output, target)

            _, pred = torch.max(output, 1)

            loss.backward()
            opt.step()

    # ## Evaluate

    model.eval()

    for samples, target in test_loader:
        samples, target = samples.cuda(), target.cuda()
        output = model(samples).cuda()
        _, y_pred = torch.max(output, 1)

    acc = accuracy_score(y_test, y_pred.cpu())
    f1 = f1_score(y_test, y_pred.cpu())
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred.cpu()))
    tpr = conf_matrix.iloc[1, 1] / (conf_matrix.iloc[1, 1] + conf_matrix.iloc[1, 0])
    fpr = conf_matrix.iloc[0, 1] / (conf_matrix.iloc[0, 1] + conf_matrix.iloc[0, 0])

    acc_arrr.append(acc)
    f1_arr.append(f1)
    tpr_arr.append(tpr)
    fpr_arr.append(fpr)
    conf_matrix_arr.append(conf_matrix)

    print('Accuracy: %0.4f' % acc)
    print('F1 Score: %0.4f' % f1)
    print('TPR: %0.4f' % tpr)
    print('FPR: %0.4f' % fpr)
    print('\nConfusion Matrix:')
    print(conf_matrix)  # rows are the true label, columns are the predicted label ([0,1] is FP, [1,0] is FN)
    print()

# Get average, median, and standard deviation for confusion matrix
tn = []
tp = []
fn = []
fp = []

for df in conf_matrix_arr:
    tn.append(conf_matrix.iloc[0, 0])
    tp.append(conf_matrix.iloc[1, 1])
    fn.append(conf_matrix.iloc[1, 0])
    fp.append(conf_matrix.iloc[0, 1])

conf_matrix_avg = pd.DataFrame([[np.mean(tn), np.mean(fp)],[np.mean(fn), np.mean(tp)]])
conf_matrix_med = pd.DataFrame([[np.median(tn), np.median(fp)],[np.median(fn), np.median(tp)]])
conf_matrix_std = pd.DataFrame([[np.std(tn), np.std(fp)],[np.std(fn), np.std(tp)]])

print('average accuracy: %0.4f' % np.mean(acc_arr))
print('average F1: %0.4f' % np.mean(f1_arr))
print('average TPR: %0.4f' % np.mean(tpr_arr))
print('average FPR: %0.4f' % np.mean(fpr_arr))
print('average accuracy: %0.4f' % np.mean(acc_arr))
print('average confusion matrix')
print(conf_matrix_avg)
print()

print('median accuracy: %0.4f' % np.median(acc_arr))
print('median F1: %0.4f' % np.median(f1_arr))
print('median TPR: %0.4f' % np.median(tpr_arr))
print('median FPR: %0.4f' % np.median(fpr_arr))
print('median accuracy: %0.4f' % np.median(acc_arr))
print('median confusion matrix')
print(conf_matrix_med)
print()

print('std accuracy: %0.4f' % np.std(acc_arr))
print('std F1: %0.4f' % np.std(f1_arr))
print('std TPR: %0.4f' % np.std(tpr_arr))
print('std FPR: %0.4f' % np.std(fpr_arr))
print('std accuracy: %0.4f' % np.std(acc_arr))
print('std confusion matrix')
print(conf_matrix_std)
print()

# Save data
np.savez_compressed('results/'run+'/res.npz',
    acc=acc_arr,
    f1=f1_arr,
    tpr=tpr_arr,
    fpr=fpr_arr,
    conf_matrix=conf_matrix_arr
)
