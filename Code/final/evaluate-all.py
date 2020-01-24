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

print('\n############### Evaluate All Models ###############')

num_iter = int(sys.argv[1])         # number of iterations for splitting data

# ## Read in Data

train_test = pd.read_json('data/train-test.json')
labels = train_test[['label']]
train_test = train_test.drop('label', axis='columns')

train_size = 0.75    # train_test size is 0.8 of total dataset. want train size to be 0.6 of total dataset. so train size here is 0.75 (0.8*0.75=0.6)
base_train_size = 0.66667   # train set size of base classifiers for stacked ensemble is 0.8*0.75*0.66667 = 0.4 of total data set 

rfc_acc_arr = []
rfc_f1_arr = []
rfc_tpr_arr = []
rfc_fpr_arr = []
rfc_conf_matrix_arr = []

log_acc_arr = []
log_f1_arr = []
log_tpr_arr = []
log_fpr_arr = []
log_conf_matrix_arr = []

knn_acc_arr = []
knn_f1_arr = []
knn_tpr_arr = []
knn_fpr_arr = []
knn_conf_matrix_arr = []

meta_acc_arr = []
meta_f1_arr = []
meta_tpr_arr = []
meta_fpr_arr = []
meta_conf_matrix_arr = []

mlp_acc_arr = []
mlp_f1_arr = []
mlp_tpr_arr = []
mlp_fpr_arr = []
mlp_conf_matrix_arr = []

ab_acc_arr = []
ab_f1_arr = []
ab_tpr_arr = []
ab_fpr_arr = []
ab_conf_matrix_arr = []

for i in range(num_iter):
    # ### Split Data

    X_train, X_test, y_train, y_test = train_test_split(
        train_test.values,
        labels.values.ravel(),
        train_size=train_size,
        shuffle=True,
        random_state=None,    # Want to observe average accuracy metrics across random splits
        stratify=labels.values.ravel()
    )
 
    # ### Impute Data
    imp = IterativeImputer(max_iter=100, random_state=1337)

    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ## Define Models

#    rfc = RandomForestClassifier(
#        n_estimators=50,
#        max_depth=4,
#        criterion='entropy',
#        class_weight=None,
#        max_features='auto',
#        bootstrap=True,
#        random_state=r
#    )

    rfc = RandomForestClassifier(    # best for tpr
        n_estimators=5,
        max_depth=2,
        criterion='gini',
        class_weight='balanced',
        max_features='auto',
        bootstrap=True,
        random_state=r
    )

#    log = LogisticRegression(
#        penalty='l1',
#        class_weight=None,
#        C=1,
#        fit_intercept=True,
#        solver='liblinear',
#        random_state=r
#    )

    log = LogisticRegression(    # best for tpr
        penalty='l2',
        class_weight='balanced',
        C=1,
        fit_intercept=True,
        solver='liblinear',
        random_state=r
    )

#    knn = KNeighborsClassifier(   # knn has no random_state
#        n_neighbors=10,
#        weights='distance',
#        p=2
#    )

    knn = KNeighborsClassifier(   # knn has no random_state, best for tpr
        n_neighbors=1,
        weights='uniform',
        p=2
    )

#    rfc_base = RandomForestClassifier(
#        n_estimators=50,
#        max_depth=4,
#        criterion='entropy',
#        class_weight=None,
#        max_features='auto',
#        bootstrap=True,
#        random_state=r
#    )

    rfc_base = RandomForestClassifier(    # best for tpr
        n_estimators=5,
        max_depth=2,
        criterion='gini',
        class_weight='balanced',
        max_features='auto',
        bootstrap=True,
        random_state=r
    )

#    log_base = LogisticRegression(
#        penalty='l1',
#        class_weight=None,
#        C=1,
#        fit_intercept=True,
#        solver='liblinear',
#        random_state=r
#    )

    log_base = LogisticRegression(    # best for tpr
        penalty='l2',
        class_weight='balanced',
        C=1,
        fit_intercept=True,
        solver='liblinear',
        random_state=r
    )

#    knn_base = KNeighborsClassifier(   # knn has no random_state
#        n_neighbors=10,
#        weights='distance',
#        p=2
#    )

    knn_base = KNeighborsClassifier(   # knn has no random_state, best for tpr
        n_neighbors=1,
        weights='uniform',
        p=2
    )

#    meta = LogisticRegression(   # logistic regression for stacked ensemble
#        penalty='l1',
#        class_weight='balanced',
#        C=50,
#        fit_intercept=True,
#        solver='liblinear',
#        random_state=r
#    )

    meta = LogisticRegression(   # logistic regression for stacked ensemble, best for tpr
        penalty='l2',
        class_weight='balanced',
        C=3,
        fit_intercept=True,
        solver='liblinear',
        random_state=r
    )

#    mlp = MLPClassifier(
#        hidden_layer_sizes=(50, 100, 50),
#        solver='sgd',
#        max_iter=200,
#        batch_size=16,
#        learning_rate='adaptive',
#        learning_rate_init=0.1,
#        random_state=r,
#        momentum=0.9,
#        nesterovs_momentum=False
#    )

    mlp = MLPClassifier(        # best for tpr
        hidden_layer_sizes=(200, 200, 200),
        solver='sgd',
        max_iter=200,
        batch_size=16,
        learning_rate='adaptive',
        learning_rate_init=0.1,
        random_state=r,
        momentum=0.9,
        nesterovs_momentum=False
    )

    dtc = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=3,
        max_features=None,
        class_weight='balanced',
        random_state=r
    )

#    ab = AdaBoostClassifier(
#        base_estimator=dtc,
#        n_estimators=100,
#        learning_rate=0.7,
#        random_state=r
#    )

    ab = AdaBoostClassifier(    # best for tpr
        base_estimator=dtc,
        n_estimators=5,
        learning_rate=0.1,
        random_state=r
    )

    # ## Evaluate

    # ### rfc

    rfc.fit(X_train, y_train)
    rfc_y_pred = rfc.predict(X_test)

    rfc_acc = accuracy_score(y_test, rfc_y_pred)
    rfc_f1 = f1_score(y_test, rfc_y_pred)
    rfc_conf_matrix = pd.DataFrame(confusion_matrix(y_test, rfc_y_pred))
    rfc_tpr = rfc_conf_matrix.iloc[1, 1] / (rfc_conf_matrix.iloc[1, 1] + rfc_conf_matrix.iloc[1, 0])
    rfc_fpr = rfc_conf_matrix.iloc[0, 1] / (rfc_conf_matrix.iloc[0, 1] + rfc_conf_matrix.iloc[0, 0])

    rfc_acc_arr.append(rfc_acc)
    rfc_f1_arr.append(rfc_f1)
    rfc_tpr_arr.append(rfc_tpr)
    rfc_fpr_arr.append(rfc_fpr)
    rfc_conf_matrix_arr.append(rfc_conf_matrix)

    # ### log

    log.fit(X_train, y_train)
    log_y_pred = log.predict(X_test)

    log_acc = accuracy_score(y_test, log_y_pred)
    log_f1 = f1_score(y_test, log_y_pred)
    log_conf_matrix = pd.DataFrame(confusion_matrix(y_test, log_y_pred))
    log_tpr = log_conf_matrix.iloc[1, 1] / (log_conf_matrix.iloc[1, 1] + log_conf_matrix.iloc[1, 0])
    log_fpr = log_conf_matrix.iloc[0, 1] / (log_conf_matrix.iloc[0, 1] + log_conf_matrix.iloc[0, 0])

    log_acc_arr.append(log_acc)
    log_f1_arr.append(log_f1)
    log_tpr_arr.append(log_tpr)
    log_fpr_arr.append(log_fpr)
    log_conf_matrix_arr.append(log_conf_matrix)

    # ### knn

    knn.fit(X_train, y_train)
    knn_y_pred = knn.predict(X_test)

    knn_acc = accuracy_score(y_test, knn_y_pred)
    knn_f1 = f1_score(y_test, knn_y_pred)
    knn_conf_matrix = pd.DataFrame(confusion_matrix(y_test, knn_y_pred))
    knn_tpr = knn_conf_matrix.iloc[1, 1] / (knn_conf_matrix.iloc[1, 1] + knn_conf_matrix.iloc[1, 0])
    knn_fpr = knn_conf_matrix.iloc[0, 1] / (knn_conf_matrix.iloc[0, 1] + knn_conf_matrix.iloc[0, 0])

    knn_acc_arr.append(knn_acc)
    knn_f1_arr.append(knn_f1)
    knn_tpr_arr.append(knn_tpr)
    knn_fpr_arr.append(knn_fpr)
    knn_conf_matrix_arr.append(knn_conf_matrix)

    # ### mlp

    mlp.fit(X_train, y_train)
    mlp_y_pred = mlp.predict(X_test)

    mlp_acc = accuracy_score(y_test, mlp_y_pred)
    mlp_f1 = f1_score(y_test, mlp_y_pred)
    mlp_conf_matrix = pd.DataFrame(confusion_matrix(y_test, mlp_y_pred))
    mlp_tpr = mlp_conf_matrix.iloc[1, 1] / (mlp_conf_matrix.iloc[1, 1] + mlp_conf_matrix.iloc[1, 0])
    mlp_fpr = mlp_conf_matrix.iloc[0, 1] / (mlp_conf_matrix.iloc[0, 1] + mlp_conf_matrix.iloc[0, 0])

    mlp_acc_arr.append(mlp_acc)
    mlp_f1_arr.append(mlp_f1)
    mlp_tpr_arr.append(mlp_tpr)
    mlp_fpr_arr.append(mlp_fpr)
    mlp_conf_matrix_arr.append(mlp_conf_matrix)

    # ### ab

    ab.fit(X_train, y_train)
    ab_y_pred = ab.predict(X_test)

    ab_acc = accuracy_score(y_test, ab_y_pred)
    ab_f1 = f1_score(y_test, ab_y_pred)
    ab_conf_matrix = pd.DataFrame(confusion_matrix(y_test, ab_y_pred))
    ab_tpr = ab_conf_matrix.iloc[1, 1] / (ab_conf_matrix.iloc[1, 1] + ab_conf_matrix.iloc[1, 0])
    ab_fpr = ab_conf_matrix.iloc[0, 1] / (ab_conf_matrix.iloc[0, 1] + ab_conf_matrix.iloc[0, 0])

    ab_acc_arr.append(ab_acc)
    ab_f1_arr.append(ab_f1)
    ab_tpr_arr.append(ab_tpr)
    ab_fpr_arr.append(ab_fpr)
    ab_conf_matrix_arr.append(ab_conf_matrix)

    # ### meta

    # #### Train Meta-Classifier

    train_base_len = int(len(X_train) * base_train_size)
    X_train_base = X_train[:train_base_len]    # training set of base classifiers
    y_train_base = y_train[:train_base_len]

    X_test_base = X_train[train_base_len:]    # output of testing set for base classifiers becomes training set of meta classifier
    y_test_base = y_train[train_base_len:]

    X_test_meta = X_test    # output of this test set by base classifiers becomes test set of meta classifier
    y_test_meta = y_test

    rfc_base.fit(X_train_base, y_train_base)
    rfc_base_prob = rfc_base.predict_proba(X_test_base)[:, 1]
    log_base.fit(X_train_base, y_train_base)
    log_base_prob = log_base.predict_proba(X_test_base)[:, 1]
    knn_base.fit(X_train_base, y_train_base)
    knn_base_prob = log_base.predict_proba(X_test_base)[:, 1]

    X_train_meta = np.hstack((
        log_base_prob.reshape(len(log_base_prob), 1), 
        rfc_base_prob.reshape(len(rfc_base_prob), 1), 
        knn_base_prob.reshape(len(knn_base_prob), 1), 
        log_base_prob.reshape(len(log_base_prob), 1)*rfc_base_prob.reshape(len(rfc_base_prob), 1), 
        log_base_prob.reshape(len(log_base_prob), 1)*knn_base_prob.reshape(len(knn_base_prob), 1), 
        rfc_base_prob.reshape(len(rfc_base_prob), 1)*knn_base_prob.reshape(len(knn_base_prob), 1), 
    ))
    y_train_meta = y_test_base

    meta.fit(X_train_meta, y_train_meta)

    # #### Test Meta-Classifier

    rfc_base_prob = rfc_base.predict_proba(X_test_meta)[:, 1]
    log_base_prob = log_base.predict_proba(X_test_meta)[:, 1]
    knn_base_prob = log_base.predict_proba(X_test_meta)[:, 1]

    X_test_meta = np.hstack((
        log_base_prob.reshape(len(log_base_prob), 1), 
        rfc_base_prob.reshape(len(rfc_base_prob), 1), 
        knn_base_prob.reshape(len(knn_base_prob), 1), 
        log_base_prob.reshape(len(log_base_prob), 1)*rfc_base_prob.reshape(len(rfc_base_prob), 1), 
        log_base_prob.reshape(len(log_base_prob), 1)*knn_base_prob.reshape(len(knn_base_prob), 1), 
        rfc_base_prob.reshape(len(rfc_base_prob), 1)*knn_base_prob.reshape(len(knn_base_prob), 1), 
    ))
    
    meta_y_pred = meta.predict(X_test_meta)

    meta_acc = accuracy_score(y_test_meta, meta_y_pred)
    meta_f1 = f1_score(y_test_meta, meta_y_pred)
    meta_conf_matrix = pd.DataFrame(confusion_matrix(y_test_meta, meta_y_pred))
    meta_tpr = meta_conf_matrix.iloc[1, 1] / (meta_conf_matrix.iloc[1, 1] + meta_conf_matrix.iloc[1, 0])
    meta_fpr = meta_conf_matrix.iloc[0, 1] / (meta_conf_matrix.iloc[0, 1] + meta_conf_matrix.iloc[0, 0])

    meta_acc_arr.append(meta_acc)
    meta_f1_arr.append(meta_f1)
    meta_tpr_arr.append(meta_tpr)
    meta_fpr_arr.append(meta_fpr)
    meta_conf_matrix_arr.append(meta_conf_matrix)

# Get average, median, standard deviation, and confusion matrix for every model
model_names = ['RFC', 'LOG', 'KNN', 'MLP', 'DT-BOOST', 'META']
acc_arr = [rfc_acc_arr, log_acc_arr, knn_acc_arr, mlp_acc_arr, ab_acc_arr, meta_acc_arr]
f1_arr = [rfc_f1_arr, log_f1_arr, knn_f1_arr, mlp_f1_arr, ab_f1_arr, meta_f1_arr]
conf_matrix_arr = [rfc_conf_matrix_arr, log_conf_matrix_arr, knn_conf_matrix_arr, mlp_conf_matrix_arr, ab_conf_matrix_arr, ab_conf_matrix_arr]
tpr_arr = [rfc_tpr_arr, log_tpr_arr, knn_tpr_arr, mlp_tpr_arr, ab_tpr_arr, meta_tpr_arr]
fpr_arr = [rfc_fpr_arr, log_fpr_arr, knn_fpr_arr, mlp_fpr_arr, ab_fpr_arr, meta_fpr_arr]

for i in range(len(model_names)): 
    print('---------------- ' + model_names[i] + ' -----------------')
    tn = []
    tp = []
    fn = []
    fp = []
    
    for df in conf_matrix_arr[i]:
        tn.append(df.iloc[0, 0])
        tp.append(df.iloc[1, 1])
        fn.append(df.iloc[1, 0])
        fp.append(df.iloc[0, 1])

    conf_matrix_avg = pd.DataFrame([[np.mean(tn), np.mean(fp)],[np.mean(fn), np.mean(tp)]])
    conf_matrix_med = pd.DataFrame([[np.median(tn), np.median(fp)],[np.median(fn), np.median(tp)]])
    conf_matrix_std = pd.DataFrame([[np.std(tn), np.std(fp)],[np.std(fn), np.std(tp)]])

    print('average accuracy: %0.4f' % np.mean(acc_arr[i]))
    print('average F1: %0.4f' % np.mean(f1_arr[i]))
    print('average TPR: %0.4f' % np.mean(tpr_arr[i]))
    print('average FPR: %0.4f' % np.mean(fpr_arr[i]))
    print('average confusion matrix')
    print(conf_matrix_avg)
    print()

    print('median accuracy: %0.4f' % np.median(acc_arr[i]))
    print('median F1: %0.4f' % np.median(f1_arr[i]))
    print('median TPR: %0.4f' % np.median(tpr_arr[i]))
    print('median FPR: %0.4f' % np.median(fpr_arr[i]))
    print('median confusion matrix')
    print(conf_matrix_med)
    print()

    print('std accuracy: %0.4f' % np.std(acc_arr[i]))
    print('std F1: %0.4f' % np.std(f1_arr[i]))
    print('std TPR: %0.4f' % np.std(tpr_arr[i]))
    print('std FPR: %0.4f' % np.std(fpr_arr[i]))
    print('std confusion matrix')
    print(conf_matrix_std)
    print()

    # Save data
    np.savez_compressed('./output/eval-all.npz',
        acc=acc_arr,
        f1=f1_arr,
        tpr=tpr_arr,
        fpr=fpr_arr,
        conf_matrix=conf_matrix_arr
    )
