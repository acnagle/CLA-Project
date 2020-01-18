#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

import copy
import sys
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# np.random.seed(0)

print('\n############### Stack ML Models ###############')

run = sys.argv[1]
num_iter = int(sys.argv[2])

# ## Read in Data

data = pd.read_json('../data.json')
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

train_size = 0.7

df = data[keep]
# df = df[df.index > '2016']   # only keep data after 2015
# labels = labels.loc[df.index]

acc_arr = []
f1_arr = []
tpr_arr = []
fpr_arr = []
conf_matrix_arr = []

train_test_size = 0.8
train_size = 0.625
data_aug = False
batch_size = 16
rand_state = None  #1337

for i in range(num_iter):
    print('Iteration', i+1)

    train_test_idx, hold_idx, y_train_test, y_hold = train_test_split(
        np.arange(len(df)),
        labels.values.ravel(),
        train_size=train_test_size,
        shuffle=True,
        stratify=labels.values.ravel(),
        random_state=rand_state
    )

    X_train_test = df.iloc[train_test_idx].values
    X_hold = df.iloc[hold_idx].values

    train_idx, test_idx, y_train, y_test = train_test_split(
        train_test_idx,
        y_train_test,
        train_size=train_size,
        shuffle=True,
        stratify=y_train_test,
        random_state=rand_state
    )

    X_train = df.iloc[train_idx].values
    X_test = df.iloc[test_idx].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_hold = scaler.transform(X_hold)

    # ## Define Base Learners

    log = LogisticRegression(
        penalty='l1',
        tol=0.0001,
        C=1,
        fit_intercept=True,
        class_weight='balanced',
        solver='liblinear'
    )

    rfc = RandomForestClassifier(
        n_estimators=10,
        max_depth=4,
        criterion='gini',
        bootstrap=True,
        class_weight='balanced_subsample'
    )

    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',    # or distance
        p=2
    )

    # ## Evaluate

    log.fit(X_train, y_train)
    log_y_prob = log.predict_proba(X_test)[:, 1]
    
    rfc.fit(X_train, y_train)
    rfc_y_prob = rfc.predict_proba(X_test)[:, 1]

    knn.fit(X_train, y_train)
    knn_y_prob = rfc.predict_proba(X_test)[:, 1]

    # ## Create Training Set for Meta-Learner
    X_train_meta = np.hstack((
        log_y_prob.reshape(len(log_y_prob), 1),
        rfc_y_prob.reshape(len(rfc_y_prob), 1),
        knn_y_prob.reshape(len(knn_y_prob), 1),
    ))
    y_train_meta = y_test

    # ## Define Meta-Learner
    meta = RandomForestClassifier(
        n_estimators=10,
        max_depth=4,
        criterion='gini',
        bootstrap=True,
        class_weight='balanced_subsample'
    )

    # ## Evaluate
    log_y_hold_pred = log.predict(X_hold)
    log_y_hold_prob = log.predict_proba(X_hold)[:, 1]
    rfc_y_hold_pred = log.predict(X_hold)
    rfc_y_hold_prob = log.predict_proba(X_hold)[:, 1]
    knn_y_hold_pred = log.predict(X_hold)
    knn_y_hold_prob = log.predict_proba(X_hold)[:, 1]

    X_hold_meta = np.hstack((
        log_y_hold_prob.reshape(len(log_y_hold_prob), 1),
        rfc_y_hold_prob.reshape(len(rfc_y_hold_prob), 1),
        knn_y_hold_prob.reshape(len(knn_y_hold_prob), 1),
    ))
    y_hold_meta = y_hold

    meta.fit(X_train_meta, y_train_meta)
    y_pred = meta.predict(X_hold_meta)

    acc = accuracy_score(y_hold, y_pred)
    f1 = f1_score(y_hold, y_pred)
    conf_matrix = pd.DataFrame(confusion_matrix(y_hold, y_pred))
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

    coef_sort_idx = np.argsort(-np.abs(meta.feature_importances_), kind='mergesort')

    print('Feature weighting for random forests\n')
    for idx in coef_sort_idx:
        coef = meta.feature_importances_[idx]
        
        if coef < 0:
            print('\t%0.4f' % meta.feature_importances_[idx], df.columns[idx])
        else:
            print('\t %0.4f' % meta.feature_importances_[idx], df.columns[idx])

    print('-'*15)
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
np.savez_compressed('results/'+run+'/stack.npz',
    acc=acc_arr,
    f1=f1_arr,
    tpr=tpr_arr,
    fpr=fpr_arr,
    conf_matrix=conf_matrix_arr
)
