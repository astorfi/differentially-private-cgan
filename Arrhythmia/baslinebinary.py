### https://www.kaggle.com/shayanfazeli/heartbeat?select=mitbih_train.csv

import pandas as pd
import sklearn.model_selection as skl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np
from random import sample
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
DATASETDIR = os.path.expanduser('~/data/Arrhythmia')
train = pd.read_csv(os.path.join(DATASETDIR, 'mitbih_train.csv'), sep=',').to_numpy()
test = pd.read_csv(os.path.join(DATASETDIR, 'mitbih_test.csv'), sep=',').to_numpy()

# Distribution of classes
unique, unique_counts = np.unique(train[:,-1], return_counts=True)
print('Unique classes', unique)
print('Count of unique classes', unique_counts)

# Make things binary
def make_binary(data):
    labels = data[:,-1]
    labels[labels!=0] = 1
    data[:,-1] = labels
    return data

train = make_binary(train)
test = make_binary(test)

# Report positive labels distribution
print('Positive label percentage in train set', train[:,-1].sum()/train.shape[0])
print('Positive label percentage in test set', test[:,-1].sum()/test.shape[0])

# Data shape
print('train shape', train.shape)
print('test shape', test.shape)

# feature size
print('num features: ', train.shape[1])

# Associated features
X_train, y_train, X_test, y_test = train[:,:-1], train[:,-1], test[:,:-1], test[:,-1]

### Effect of noise on classification
mu, sigma = 0, 0.1 # mean and standard deviation
noise = np.random.normal(mu, sigma, (X_train.shape[0],X_train.shape[1]))
X_train += noise

# Scaling features
# sc = StandardScaler()
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# unified sets
train = np.concatenate((X_train, y_train[:,None]), axis=1)
test = np.concatenate((X_test, y_test[:,None]), axis=1)

# Save the processed data
pd.DataFrame(train).to_csv(os.path.join(DATASETDIR,'train.csv'))
pd.DataFrame(test).to_csv(os.path.join(DATASETDIR,'test.csv'))

###############################
######## Classifier ###########
###############################

# Supervised transformation based on random forests
# Good to know about feature transformation
n_estimator=100
cls = RandomForestClassifier(max_depth=5, n_estimators=n_estimator)
# cls = GradientBoostingClassifier(n_estimators=n_estimator)
cls.fit(X_train, y_train)
y_pred = cls.predict_proba(X_test)[:, 1]

# ROC
fpr_rf_lm, tpr_rf_lm, _ = metrics.roc_curve(y_test, y_pred)
print('AUROC: ', metrics.auc(fpr_rf_lm,tpr_rf_lm))

# PR
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
AUPRC = metrics.auc(recall, precision)
print('AP: ', metrics.average_precision_score(y_test, y_pred))
print('Area under the precision recall curve: ', AUPRC)

print("confusion matrix:")
y_pred = cls.predict(X_test)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cf_matrix)
ax = sns.heatmap(cf_matrix, annot=True)
plt.show()
