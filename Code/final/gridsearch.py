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

print('\n############### Grid Search ###############')

# ## Read in Data

train_test = pd.read_json('data/train-test.json')
labels = train_test[['label']]
train_test = train_test.drop('label', axis='columns')

train_size = 0.75   # train_test size is 0.8 of total dataset. want train size to be 0.6 of total dataset. so train size here is 0.75 (0.8*0.75=0.6)

# ### Split Data

X_train, X_test, y_train, y_test = train_test_split(
    train_test.values,
    labels.values.ravel(),
    train_size=train_size,
    random_state=r,
    shuffle=True,
    stratify=labels.values.ravel()
)

# ### Impute Data
imp = IterativeImputer(max_iter=100, random_state=r)

X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

# ### Augment Data
#smote = SMOTE(
#    sampling_strategy='all',
#    random_state=1337,
#    k_neighbors=5,
#    n_jobs=1
#)
#
#X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ## Define models
rfc = RandomForestClassifier(
    max_features='auto',
    bootstrap=True,
    random_state=r
)

log = LogisticRegression(
    fit_intercept=True,
    solver='liblinear',
    random_state=r
)

meta = LogisticRegression(   # logistic regression for stacked ensemble
    fit_intercept=True,
    solver='liblinear',
    random_state=r
)

knn = KNeighborsClassifier()   # knn has no random_state

mlp = MLPClassifier(
    solver='sgd',
    batch_size=16,
    learning_rate='adaptive',
    learning_rate_init=0.1,
    random_state=r,
    momentum=0.9
)

dtc = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=3,
    max_features=None,
    class_weight='balanced',
    random_state=r
)

ab = AdaBoostClassifier(
    base_estimator=dtc,
    random_state=r
)

# ## Define parameter grids
rfc_params = {
    'n_estimators': [5, 10, 20, 50, 70, 100],
    'criterion': ('gini', 'entropy'),
    'max_depth': [2, 3, 4, 5],
    'class_weight': ('balanced', 'balanced_subsample', None)
}

log_params = {
    'penalty': ('l1', 'l2'),
    'C': [1, 1.5, 2, 3, 4, 5, 10, 20, 50, 100],
    'class_weight': ('balanced', None)
}

knn_params = {
    'n_neighbors': [1, 2, 3, 4, 5, 10],
    'weights': ('uniform', 'distance'),
    'p': [1, 2]
}

mlp_params = {
    'hidden_layer_sizes': [(100), (50, 100, 50), (100, 100, 100), (100, 200, 100), (200, 200, 200), (300, 300, 300)],
    'max_iter': [200, 250, 300],
    'nesterovs_momentum': [True, False],
}

ab_params = {
    'n_estimators': [5, 10, 20, 50, 70, 100],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1],
}

# ## Define Searches

refit = 'recall'   # or 'accuracy'

rfc_grid = GridSearchCV(
    rfc,
    rfc_params,
    scoring=['accuracy', 'recall'],
    n_jobs=8,
    cv=5,
    refit=refit
)

log_grid = GridSearchCV(
    log,
    log_params,
    scoring=['accuracy', 'recall'],
    n_jobs=8,
    cv=5,
    refit=refit
)


meta_grid = GridSearchCV(
    meta,
    log_params,
    scoring=['accuracy', 'recall'],
    n_jobs=8,
    cv=5,
    refit=refit
)

knn_grid = GridSearchCV(
    knn,
    knn_params,
    scoring=['accuracy', 'recall'],
    n_jobs=8,
    cv=5,
    refit=refit
)

mlp_grid = GridSearchCV(
    mlp,
    mlp_params,
    scoring=['accuracy', 'recall'],
    n_jobs=8,
    cv=5,
    refit=refit
)

ab_grid = GridSearchCV(
    ab,
    ab_params,
    scoring=['accuracy', 'recall'],
    n_jobs=8,
    cv=5,
    refit=refit
)

# ## Report Best Parameters for each Model

grids = {'RFC': rfc_grid, 'LOG': log_grid, 'KNN': knn_grid, 'MLP': mlp_grid, 'DT-BOOST': ab_grid}

for key in list(grids.keys()):
    grids[key].fit(X_train, y_train)

    print('---------------- ' + key + ' ----------------')
    print('Grid Search CV Results:')
#    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#        print(pd.DataFrame.from_dict(grids[key].cv_results_))

    print('\nBest Parameters:')
    print(grids[key].best_params_)

    print('\nBest Score', grids[key].best_score_)
    print('\n\n\n')

# ## Create Stacked Ensemble Training Set
# Use best rfc, knn, log models determined above as base classifiers
rfc_base = rfc_grid.best_estimator_
knn_base = knn_grid.best_estimator_
log_base = log_grid.best_estimator_

log_y_prob = log_base.predict_proba(X_test)[:, 1]
rfc_y_prob = rfc_base.predict_proba(X_test)[:, 1]
knn_y_prob = knn_base.predict_proba(X_test)[:, 1]

# ## Create Training Set for Meta-Learner
X_train_meta = np.hstack((
    log_y_prob.reshape(len(log_y_prob), 1),
    rfc_y_prob.reshape(len(rfc_y_prob), 1),
    knn_y_prob.reshape(len(knn_y_prob), 1),
    log_y_prob.reshape(len(log_y_prob), 1)*rfc_y_prob.reshape(len(rfc_y_prob), 1),
    log_y_prob.reshape(len(log_y_prob), 1)*knn_y_prob.reshape(len(knn_y_prob), 1),
    rfc_y_prob.reshape(len(rfc_y_prob), 1)*knn_y_prob.reshape(len(knn_y_prob), 1),
))
y_train_meta = y_test

meta_scaler = StandardScaler()
X_train_meta = scaler.fit_transform(X_train_meta)

meta_grid.fit(X_train_meta, y_train_meta)

print('---------------- STACK  ----------------')
print('Grid Search CV Results:')
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(pd.DataFrame.from_dict(meta_grid.cv_results_))

print('\nBest Parameters:')
print(meta_grid.best_params_)

print('\nBest Score', meta_grid.best_score_)
print()

