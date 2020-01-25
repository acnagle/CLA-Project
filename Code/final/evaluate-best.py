#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

import random
import copy
import sys
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE

r = 1337    # random seed
np.random.seed(r)
random.seed(r)

print('\n############### Evaluate Best Model ###############')

# ## Read in Data

train_test = pd.read_json('data/train-test.json')
train_test_labels = train_test[['label']]
train_test = train_test.drop('label', axis='columns')

hold = pd.read_json('data/hold.json')
hold_labels = hold[['label']]
hold = hold.drop('label', axis='columns')

# ### Impute Data
imp = IterativeImputer(max_iter=100, random_state=r)

X_train_test = imp.fit_transform(train_test.values)
y_train_test = train_test_labels.values.ravel()

X_hold = imp.transform(hold.values)
y_hold = hold_labels.values.ravel()

# ### Augment Data
#if smote_ratio > 0:
#    smote = SMOTE(
#                sampling_strategy='all',
#                random_state=1337,
#                k_neighbors=5,
#                n_jobs=1
#            )
#
#    X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_test = scaler.fit_transform(X_train_test)
X_hold = scaler.transform(X_hold)

# ## Define Models

best = RandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    criterion='entropy',
    class_weight=None,
    max_features='auto',
    bootstrap=True,
    random_state=r
)

# ## Evaluate

best.fit(X_train_test, y_train_test)
y_pred = rfc.predict(X_train_test)

acc = accuracy_score(y_hold, y_pred)
f1 = f1_score(y_hold, y_pred)
conf_matrix = pd.DataFrame(confusion_matrix(y_hold, y_pred))
tpr = conf_matrix.iloc[1, 1] / (conf_matrix.iloc[1, 1] + conf_matrix.iloc[1, 0])
fpr = conf_matrix.iloc[0, 1] / (conf_matrix.iloc[0, 1] + conf_matrix.iloc[0, 0])

print('Holdout Accuracy: %0.4f' % np.mean(acc_arr[i]))
print('Holdout F1: %0.4f' % np.mean(f1_arr[i]))
print('Holdout TPR: %0.4f' % np.mean(tpr_arr[i]))
print('Holdout FPR: %0.4f' % np.mean(fpr_arr[i]))
print('Holdout confusion matrix')
print(conf_matrix)
print()

