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

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None
# np.random.seed(0)

# ## Read in Data

data = pd.read_json('data.json')

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    display(data[['label', 'algalBloomSheen_one_day', 'algalBloomSheen_three_day', 'algalBloomSheen_one_week']])

labels = data[['label']]
data = data.drop('label', axis='columns')

# ## Add Features

data['log_turbidity'] = np.log(data['turbidity'] + 1)

# ## Correlation

corr = data.corr()

corr_thresh = 0.80  # threshold for correlation. for any two variables with correlation > thresh, one is removed

thresh = corr.abs() > corr_thresh

keep = copy.deepcopy(data.columns).to_list()

print('\nRemoved features: ')
# keep features whose correlation with other features is <= corr_thresh
for i in range(0, len(thresh.index)):
    for j in range(i+1, len(thresh.columns)):
        if thresh.iloc[i, j]:
            if thresh.columns[j] in keep:
                print('\t', thresh.columns[j])
                keep.remove(thresh.columns[j])
print()

# handpicked keep based on results above
# keep = ['turbidity', 'rel_hum', 'wind_speed', 'chlor',
#        'phycocyanin', 'do_sat', 'do_wtemp', 'pco2_ppm', 'par',
#        'par_below', 'DAILYMaximumDryBulbTemp', 'DAILYMinimumDryBulbTemp',
#        'DAILYPrecip',
#        'DAILYAverageStationPressure',
#        'cos_month', 'sin_month', 'cos_wind_dir', 'sin_wind_dir',
#        'DAILYPrecip_one_day', 'DAILYPrecip_three_day', 'DAILYPrecip_one_week',
#        'algalBloomSheen_one_week']

df = data[keep]

# ## Prepare Data/Create Dataloaders

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

train_test_size = 0.80
train_size = 0.75
batch_size = 16
data_aug = False    # data augmentation

# zero pad data set. The input format for the resnet must be 5x5, or 6x6, or 7x7, etc.
pad_df = copy.deepcopy(df)
for i in range(36-df.shape[1]):
    pad_df[i] = [1 for _ in range(df.shape[0])]

vals = pad_df.values
data_reshape = []
for i in range(pad_df.shape[0]):
    data_reshape.append(vals[i].reshape(6, 6))

# Stratified split into training and holdout set
X_train_test, X_hold, y_train_test, y_hold = train_test_split(
    data_reshape,
    labels.values.ravel(),
    train_size=train_test_size,    # 80% of the data set
    shuffle=True,
    stratify=labels.values.ravel()
)

# Stratified split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_train_test,
    y_train_test,
    train_size=train_size,    # final size of training set: 75%*80%=60%
    shuffle=True,
    stratify=y_train_test
)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_hold = np.asarray(X_hold)

print('Training set size:', X_train.shape)
print('Testing set size:', X_test.shape)
print('Holdout set size:', X_hold.shape)

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
train_test_set = AlgalBloomDataset(X_train_test, y_train_test, trnsfrm)

test_set = AlgalBloomDataset(X_test, y_test, 
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229]),
    ])
)

hold_set = AlgalBloomDataset(X_hold, y_hold, 
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229]),
    ])
)

train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = data_utils.DataLoader(test_set, batch_size=len(test_set), shuffle=True)
train_test_loader = data_utils.DataLoader(train_test_set, batch_size=batch_size, shuffle=True)
hold_loader = data_utils.DataLoader(hold_set, batch_size=len(hold_set), shuffle=True)

# ## Define ResNet Model

learning_rate = 0.1
num_epochs = 450
weighted = False

if weighted:
    class_weights = torch.Tensor([np.bincount(y_train.astype(int))[0] / len(y_train), 
                            np.bincount(y_train.astype(int))[1] / len(y_train)])
else:
    class_weights=None

model = resnet.ResNet9().cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
opt = optim.SGD(model.parameters(), lr=learning_rate, nesterov=False, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[75, 150, 225, 300, 375], gamma=0.1)

# ## Train ResNet

loss_arr = []
train_acc_arr = []
train_f1_arr = []
test_acc_arr = []
test_f1_arr = []

for epoch in range(num_epochs):
    model.train()   # train model
    
    print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    
    epoch_loss_arr = []
    epoch_acc_arr = []
    epoch_f1_arr = []
    
    for samples, target in train_loader:
        opt.zero_grad()
        samples, target = samples.cuda(), target.cuda()
        output = model(samples)
        loss = criterion(output, target)
        
        _, pred = torch.max(output, 1)
                
        loss.backward()
        opt.step()
        
        epoch_loss_arr.append(loss.item())
        epoch_acc_arr.append(torch.sum(pred == target).float() / len(target))
        epoch_f1_arr.append(f1_score(target.cpu(), pred.cpu()))
    
    epoch_loss = sum(epoch_loss_arr)
    epoch_acc = sum(epoch_acc_arr) / len(epoch_acc_arr)
    epoch_f1 = sum(epoch_f1_arr) / len(epoch_f1_arr)
    
    loss_arr.append(epoch_loss)
    train_acc_arr.append(epoch_acc)
    train_f1_arr.append(epoch_f1)
    
    print('Training: Loss: {:0.4f} Acc: {:0.4f} F1: {:0.4f}\n'.format(epoch_loss, epoch_acc, epoch_f1))
    
    # Evaluate on test set
    model.eval()
    
    for samples, target in test_loader:
        samples, target = samples.cuda(), target.cuda()
        output = model(samples).cuda()
        _, pred = torch.max(output, 1)
        
    test_acc_arr.append(torch.sum(pred == target).float() / len(target))
    test_f1_arr.append(f1_score(target.cpu(), pred.cpu()))
    
    print('Testing:               ' + 
          'Acc: {:0.4f} F1: {:0.4f}\n'.format(test_acc_arr[epoch-1], test_f1_arr[epoch-1]))
    
# ## Plot Training Performance

x = [i for i in range(1, num_epochs+1)]

plt.figure(figsize=(12, 8))
plt.plot(x, loss_arr)
plt.xlabel('Number of Epochs')
plt.ylabel('Training Loss')
plt.title('ResNet Training Loss')
plt.xlim([1, num_epochs])
plt.grid(True)
plt.savefig('train_loss.png')

plt.figure(figsize=(12, 8))
plt.plot(x, train_acc_arr)
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.title('ResNet Training Accuracy')
plt.xlim([1, num_epochs])
plt.grid(True)
plt.savefig('train_acc.png')

plt.figure(figsize=(12, 8))
plt.plot(x, train_f1_arr)
plt.xlabel('Number of Epochs')
plt.ylabel('Training F1 Score')
plt.title('ResNet Training F1 Score')
plt.xlim([1, num_epochs])
plt.grid(True)
plt.savefig('train_f1.png')

plt.figure(figsize=(12, 8))
plt.plot(x, test_acc_arr)
plt.xlabel('Number of Epochs')
plt.ylabel('Testing Accuracy')
plt.title('ResNet Testing Accuracy')
plt.xlim([1, num_epochs])
plt.grid(True)
plt.savefig('test_acc.png')

plt.figure(figsize=(12, 8))
plt.plot(x, test_f1_arr)
plt.xlabel('Number of Epochs')
plt.ylabel('Testing F1 Score')
plt.title('ResNet Testing F1 Score')
plt.xlim([1, num_epochs])
plt.grid(True)
plt.savefig('test_f1.png')

# ## Test Model on Holdout Set

model.eval()    # test model

for samples, target in hold_loader:
    samples, target = samples.cuda(), target.cuda()
    output = model(samples).cuda()
    loss = criterion(output, target)

    _, pred = torch.max(output, 1)
    
acc = torch.sum(pred == target).float() / len(target)
f1 = f1_score(target.cpu(), pred.cpu())
conf_matrix = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), labels=[0, 1])

print('\nHoldout Accuracy: {:0.4f}'.format(acc))
print('\nF1 Score: {:0.4f}'.format(f1))
print('Confusion Matrix:')
print(pd.DataFrame(conf_matrix))
print('\n\n')

# ## Train ResNet on train_test_set

criterion = nn.CrossEntropyLoss(weight=class_weights)
opt = optim.SGD(model.parameters(), lr=learning_rate, nesterov=False, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[75, 150, 225, 300, 375], gamma=0.1)

loss_arr = []
train_test_acc_arr = []
train_test_f1_arr = []

for epoch in range(num_epochs):
    model.train()   # train model
    
    print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    
    epoch_loss_arr = []
    epoch_acc_arr = []
    epoch_f1_arr = []
    
    for samples, target in train_loader:
        opt.zero_grad()
        samples, target = samples.cuda(), target.cuda()
        output = model(samples)
        loss = criterion(output, target)
        
        _, pred = torch.max(output, 1)
                
        loss.backward()
        opt.step()
        
        epoch_loss_arr.append(loss.item())
        epoch_acc_arr.append(torch.sum(pred == target).float() / len(target))
        epoch_f1_arr.append(f1_score(target.cpu(), pred.cpu()))
    
    epoch_loss = sum(epoch_loss_arr)
    epoch_acc = sum(epoch_acc_arr) / len(epoch_acc_arr)
    epoch_f1 = sum(epoch_f1_arr) / len(epoch_f1_arr)
    
    loss_arr.append(epoch_loss)
    train_test_acc_arr.append(epoch_acc)
    train_test_f1_arr.append(epoch_f1)
    
    print('Training: Loss: {:0.4f} Acc: {:0.4f} F1: {:0.4f}\n'.format(epoch_loss, epoch_acc, epoch_f1))
    
    scheduler.step()

# ## Plot Train Test Performance

x = [i for i in range(1, num_epochs+1)]

plt.figure(figsize=(12, 8))
plt.plot(x, loss_arr)
plt.xlabel('Number of Epochs')
plt.ylabel('Train_Test Loss')
plt.title('ResNet Train_Test Loss')
plt.xlim([1, num_epochs])
plt.grid(True)
plt.savefig('train_test_loss.png')

plt.figure(figsize=(12, 8))
plt.plot(x, train_test_acc_arr)
plt.xlabel('Number of Epochs')
plt.ylabel('Train_Test Accuracy')
plt.title('ResNet Train_Test Accuracy')
plt.xlim([1, num_epochs])
plt.grid(True)
plt.savefig('train_test_acc.png')

plt.figure(figsize=(12, 8))
plt.plot(x, train_test_f1_arr)
plt.xlabel('Number of Epochs')
plt.ylabel('Train_Test F1 Score')
plt.title('ResNet Train_Test F1 Score')
plt.xlim([1, num_epochs])
plt.grid(True)
plt.savefig('train_test_f1.png')

# ## Test Model on Holdout Set

model.eval()    # test model

for samples, target in hold_loader:
    samples, target = samples.cuda(), target.cuda()
    output = model(samples).cuda()
    loss = criterion(output, target)

    _, pred = torch.max(output, 1)
    
acc = torch.sum(pred == target).float() / len(target)
f1 = f1_score(target.cpu(), pred.cpu())
conf_matrix = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), labels=[0, 1])

print('\nHoldout Accuracy: {:0.4f}'.format(acc))
print('\nF1 Score: {:0.4f}'.format(f1))
print('Confusion Matrix:')
print(pd.DataFrame(conf_matrix))
print('\n\n')



