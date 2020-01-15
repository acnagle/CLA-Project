#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

import copy
import sys
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, 
                             accuracy_score, balanced_accuracy_score)

import matplotlib.pyplot as plt

# np.random.seed(0)

run = sys.argv[1]
if not os.path.isdir('rfc/'+run+'/'):
    os.mkdir('rfc/'+run+'/')

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
# df = df[df.index > '2016']   # only keep data after 2015
# labels = labels.loc[df.index]

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

# ## Define Model

rfc = RandomForestClassifier(
    n_estimators=1000,
    max_depth=4,
    criterion='gini',
    bootstrap=True,
    class_weight='balanced'
)


# ## Evaluate

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
y_prob = rfc.predict_proba(X_test)[:, 1]

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
plt.savefig('rfc/'+run+'/roc-curve'+run+'.png')

plt.figure(figsize=(12, 8))
plt.plot(recall, precision, c='C1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('Precision Recall Curve')
plt.savefig('rfc/'+run+'/pr-curve'+run+'.png')

coef_sort_idx = np.argsort(-np.abs(rfc.feature_importances_), kind='mergesort')

print('Feature weighting for Decision Trees with AdaBoost\n')
for idx in coef_sort_idx:
    coef = rfc.feature_importances_[idx]
    
    if coef < 0:
        print('\t%0.4f' % rfc.feature_importances_[idx], df.columns[idx])
    else:
        print('\t %0.4f' % rfc.feature_importances_[idx], df.columns[idx])

# ## GridSearchCV

rfc_params = {
    'n_estimators': [10, 20, 50, 100, 200, 500, 1000, 2000],
    'criterion': ('gini', 'entropy'),
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'class_weight': ('balanced', 'balanced_subsample')
}

rfc_grid = GridSearchCV(
    estimator=rfc,
    param_grid=rfc_params,
    scoring='balanced_accuracy',    # or f1
    n_jobs=3,
    cv=5
)

rfc_grid.fit(X_train, y_train)

print('Best Random Forest Classifier:')
print(rfc_grid.best_params_)
print('Balanced Accuracy:', rfc_grid.best_score_)

model = rfc_grid.best_estimator_

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
plt.savefig('rfc/'+run+'/best-roc-curve'+run+'.png')

plt.figure(figsize=(12, 8))
plt.plot(recall, precision, c='C1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.title('Precision Recall Curve')
plt.savefig('rfc/'+run+'/best-pr-curve'+run+'.png')

coef_sort_idx = np.argsort(-np.abs(model.feature_importances_), kind='mergesort')

print('Feature weighting for Decision Trees with AdaBoost\n')
for idx in coef_sort_idx:
    coef = model.feature_importances_[idx]
    
    if coef < 0:
        print('\t%0.4f' % model.feature_importances_[idx], df.columns[idx])
    else:
        print('\t %0.4f' % model.feature_importances_[idx], df.columns[idx])

