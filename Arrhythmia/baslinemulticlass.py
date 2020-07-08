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

# Data shape
print('train shape', train.shape)
print('test shape', test.shape)

# Associated features
X_train, y_train, X_test, y_test = train[:,:-1], train[:,-1], test[:,:-1], test[:,-1]

### Effect of noise on classification
mu, sigma = 0, 1.0 # mean and standard deviation
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
import xgboost as xgb
# Supervised transformation based on random forests
# Good to know about feature transformation
n_estimator=100
# cls = RandomForestClassifier(max_depth=5, n_estimators=n_estimator)
# cls = GradientBoostingClassifier(n_estimators=n_estimator)
cls = xgb.XGBClassifier(n_estimators=n_estimator)
# cls = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
# print(np.sum(y_pred_rf==y_test) / y_test.shape[0])

score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)

c_report = metrics.classification_report(y_test, y_pred)
print('Classification report:\n', c_report)

print("confusion matrix:")
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cf_matrix)
ax = sns.heatmap(cf_matrix, annot=True)
plt.show()
