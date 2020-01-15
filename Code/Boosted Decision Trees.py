#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

import copy
import sys
import os

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, 
                             accuracy_score, balanced_accuracy_score)

import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
# np.random.seed(0)

run = sys.argv[1]
if not os.path.isdir('boosted-trees/'+run+'/'):
    os.mkdir('boosted-trees/'+run+'/')

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

train_size = 0.7

df = data[keep]

X_train, X_test, y_train, y_test = train_test_split(
    df.values,
    labels.values.ravel(),
    train_size=train_size,
    shuffle=True,
    stratify=labels.values.ravel()
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ## AdaBoost

# ### Define Model

dtc = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=3,
    max_features=None,         # or int, sqrt, log2
    class_weight='balanced'    # or none
)

ab = AdaBoostClassifier(
    base_estimator=dtc,
    n_estimators=50,
    learning_rate=0.1,
    algorithm='SAMME.R'
)

# ### Evaluate

ab.fit(X_train, y_train)
y_pred = ab.predict(X_test)
y_prob = ab.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
balanced_acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)

print('Accuracy: %0.4f' % acc)
print('Balanced Accuracy: %0.4f' % balanced_acc)
print('F1 Score: %0.4f' % f1)
print('\nConfusion Matrix:')
print(conf_matrix)  # rows are the true label, columns are the predicted label ([0,1] is FP, [1,0] is FN)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, c='C0')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('ROC Curve')
plt.savefig('boosted-trees/'+run+'/adaboost-roc-curve-'+run+'.png')

plt.figure(figsize=(12, 8))
plt.plot(recall, precision, c='C1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('Precision Recall Curve');
plt.savefig('boosted-trees/'+run+'/adaboost-pr-curve-'+run+'.png')

# ### Show Feature Importances

coef_sort_idx = np.argsort(-np.abs(ab.feature_importances_), kind='mergesort')

print('Feature weighting for Decision Trees with AdaBoost\n')
for idx in coef_sort_idx:
    coef = ab.feature_importances_[idx]
    
    if coef < 0:
        print('\t%0.4f' % ab.feature_importances_[idx], df.columns[idx])
    else:
        print('\t %0.4f' % ab.feature_importances_[idx], df.columns[idx])


# ### GridSearchCV

ab_params = {
    'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}

ab_grid = GridSearchCV(
    estimator=ab,
    param_grid=ab_params,
    scoring='accuracy',    # or f1, balanced_accuracy
    n_jobs=8,
    cv=5
)

ab_grid.fit(X_train, y_train)

print('Best AdaBoost Estimator:')
print(ab_grid.best_params_)
print('Balanced Accuracy:', ab_grid.best_score_)

model = ab_grid.best_estimator_

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
balanced_acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)

print('Accuracy: %0.4f' % acc)
print('Balanced Accuracy: %0.4f' % balanced_acc)
print('F1 Score: %0.4f' % f1)
print('\nConfusion Matrix:')
print(conf_matrix)  # rows are the true label, columns are the predicted label ([0,1] is FP, [1,0] is FN)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, c='C0')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('ROC Curve')
plt.savefig('boosted-trees/'+run+'/best-adaboost-roc-curve-'+run+'.png')

plt.figure(figsize=(12, 8))
plt.plot(recall, precision, c='C1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('Precision Recall Curve');
plt.savefig('boosted-trees/'+run+'/best-adaboost-pr-curve-'+run+'.png')

coef_sort_idx = np.argsort(-np.abs(model.feature_importances_), kind='mergesort')

print('Feature weighting for Decision Trees with AdaBoost\n')
for idx in coef_sort_idx:
    coef = model.feature_importances_[idx]
    
    if coef < 0:
        print('\t%0.4f' % model.feature_importances_[idx], df.columns[idx])
    else:
        print('\t %0.4f' % model.feature_importances_[idx], df.columns[idx])

# ## Gradient Boosting

# ### Define Model

dtc = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=3,
    max_features=None,         # or int, sqrt, log2
    class_weight='balanced'    # or none
)

gb = GradientBoostingClassifier(
    loss='deviance',
    learning_rate=0.1,
    n_estimators=100,
    subsample=1,
    criterion='friedman_mse',
    max_depth=3
)

# ### Evaluate

gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
y_prob = gb.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
balanced_acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)

print('Accuracy: %0.4f' % acc)
print('Balanced Accuracy: %0.4f' % balanced_acc)
print('F1 Score: %0.4f' % f1)
print('\nConfusion Matrix:')
print(conf_matrix)  # rows are the true label, columns are the predicted label ([0,1] is FP, [1,0] is FN)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, c='C0')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('ROC Curve')
plt.savefig('boosted-trees/'+run+'/gboost-roc-curve-'+run+'.png')

plt.figure(figsize=(12, 8))
plt.plot(recall, precision, c='C1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('Precision Recall Curve')
plt.savefig('boosted-trees/'+run+'/gboost-pr-curve-'+run+'.png')

# ### GridSearchCV

gb_params = {
    'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'subsample': [0.9, 0.95, 1],
    'max_depth': [1, 2, 3, 4, 5, 6, 7]
}

gb_grid = GridSearchCV(
    estimator=gb,
    param_grid=gb_params,
    scoring='accuracy',    # or f1, balanced_accuracy
    n_jobs=8,
    cv=5
)

gb_grid.fit(X_train, y_train)

print('Best Gradient Boost Estimator:')
print(gb_grid.best_params_)
print('Balanced Accuracy:', gb_grid.best_score_)

model = gb_grid.best_estimator_

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
balanced_acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)

print('Accuracy: %0.4f' % acc)
print('Balanced Accuracy: %0.4f' % balanced_acc)
print('F1 Score: %0.4f' % f1)
print('\nConfusion Matrix:')
print(conf_matrix)  # rows are the true label, columns are the predicted label ([0,1] is FP, [1,0] is FN)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, c='C0')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('ROC Curve')
plt.savefig('boosted-trees/'+run+'/best-gboost-roc-curve-'+run+'.png')

plt.figure(figsize=(12, 8))
plt.plot(recall, precision, c='C1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('Precision Recall Curve')
plt.savefig('boosted-trees/'+run+'/best-gboost-pr-curve-'+run+'.png')

coef_sort_idx = np.argsort(-np.abs(model.feature_importances_), kind='mergesort')

print('Feature weighting for Decision Trees with AdaBoost\n')
for idx in coef_sort_idx:
    coef = model.feature_importances_[idx]
    
    if coef < 0:
        print('\t%0.4f' % model.feature_importances_[idx], df.columns[idx])
    else:
        print('\t %0.4f' % model.feature_importances_[idx], df.columns[idx])


