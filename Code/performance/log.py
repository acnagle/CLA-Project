#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

import copy
import sys
import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE

# np.random.seed(0)

print('\n############### Logistic Regression ###############')

run = sys.argv[1]                   # run index
num_iter = int(sys.argv[2])         # number of iterations for splitting data
#num_aug = int(sys.argv[3])         # number of augmented data points to create for each data point in the data set
smote_ratio = float(sys.argv[3])    # ratio of # minority class examples divided by # majority class examples. used for SMOTE algorithm. pass 0 for no smote
data_impute = bool(sys.argv[4])     # boolean flag to indicate whether data imputation should be used

# ## Read in Data

if data_impute:
    data = pd.read_json('../data_impute.json')
else:
    data = pd.read_json('../data.json')

data = pd.read_json('../data.json')
labels = data[['label']]
data = data.drop('label', axis='columns')

# ## Add Features

# data['log_turbidity'] = np.log(data['turbidity'] + 1)

if not data_impute:
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

    df = data[keep]
else:
    df = data

#df = df[df.index > '2016']   # only keep data after 2015
#labels = labels.loc[df.index]

std = 0.1     # standard deviation of data augmentation
train_size = 0.7

acc_arr = []
f1_arr = []
tpr_arr = []
fpr_arr = []
conf_matrix_arr = []

for i in range(num_iter):
    print('Iteration', i+1)

    # ### Split Data

    X_train, X_test, y_train, y_test = train_test_split(
        df.values,
        labels.values.ravel(),
        train_size=train_size,
        shuffle=True,
        stratify=labels.values.ravel()
    )   

    # ### Impute Data
    if data_impute:
        imp = IterativeImputer(max_iter=25, random_state=1337)

        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

    # ### Augment Data
    if smote_ratio > 0:
        smote = SMOTE(
                    sampling_strategy='all',
                    random_state=1337,
                    k_neighbors=5,
                    n_jobs=1
                )   

        X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ## Define Model

    log = LogisticRegression(
        penalty='l1',
        tol=0.0001,
        C=1,
        fit_intercept=True,
        class_weight='balanced',
        solver='liblinear'
    )

    # ## Evaluate

    log.fit(X_train, y_train)
    y_pred = log.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
    tpr = conf_matrix.iloc[1, 1] / (conf_matrix.iloc[1, 1] + conf_matrix.iloc[1, 0])
    fpr = conf_matrix.iloc[0, 1] / (conf_matrix.iloc[0, 1] + conf_matrix.iloc[0, 0])

    acc_arr.append(acc)
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

    # Print feature importances
    #coef_sort_idx = np.argsort(-np.abs(log.coef_[0]), kind='mergesort')
    #
    #print('Feature weighting for logistic regression\n')
    #for idx in coef_sort_idx:
    #    coef = log.coef_[0][idx]
    #    
    #    if coef < 0:
    #        print('\t%0.4f' % log.coef_[0][idx], df.columns[idx])
    #    else:
    #        print('\t %0.4f' % log.coef_[0][idx], df.columns[idx])

    print('-'*15)
    print()


# Get average, median, and standard deviation for confusion matrix
tn = []
tp = []
fn = []
fp = []

for df in conf_matrix_arr:
    tn.append(df.iloc[0, 0])
    tp.append(df.iloc[1, 1])
    fn.append(df.iloc[1, 0])
    fp.append(df.iloc[0, 1])

conf_matrix_avg = pd.DataFrame([[np.mean(tn), np.mean(fp)],[np.mean(fn), np.mean(tp)]])
conf_matrix_med = pd.DataFrame([[np.median(tn), np.median(fp)],[np.median(fn), np.median(tp)]])
conf_matrix_std = pd.DataFrame([[np.std(tn), np.std(fp)],[np.std(fn), np.std(tp)]])

print('average accuracy: %0.4f' % np.mean(acc_arr))
print('average F1: %0.4f' % np.mean(f1_arr))
print('average TPR: %0.4f' % np.mean(tpr_arr))
print('average FPR: %0.4f' % np.mean(fpr_arr))
print('average confusion matrix')
print(conf_matrix_avg)
print()

print('median accuracy: %0.4f' % np.median(acc_arr))
print('median F1: %0.4f' % np.median(f1_arr))
print('median TPR: %0.4f' % np.median(tpr_arr))
print('median FPR: %0.4f' % np.median(fpr_arr))
print('median confusion matrix')
print(conf_matrix_med)
print()

print('std accuracy: %0.4f' % np.std(acc_arr))
print('std F1: %0.4f' % np.std(f1_arr))
print('std TPR: %0.4f' % np.std(tpr_arr))
print('std FPR: %0.4f' % np.std(fpr_arr))
print('std confusion matrix')
print(conf_matrix_std)
print()

# Save data
np.savez_compressed('results/'+run+'/log.npz',
    acc=acc_arr,
    f1=f1_arr,
    tpr=tpr_arr,
    fpr=fpr_arr,
    conf_matrix=conf_matrix_arr
)
