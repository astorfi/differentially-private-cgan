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

# Read data
DATASETDIR = os.path.expanduser('~/data/Arrhythmia')
train = pd.read_csv(os.path.join(DATASETDIR, 'mitbih_train.csv'), sep=',').to_numpy()
test = pd.read_csv(os.path.join(DATASETDIR, 'mitbih_test.csv'), sep=',').to_numpy()

# Data shape
print('train shape', train.shape)
print('test shape', test.shape)

# Associated features
X_train, y_train, X_test, y_test = train[:,:-1], train[:,-1], test[:,:-1], test[:,-1]

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
# cls = RandomForestClassifier(max_depth=5, n_estimators=n_estimator)
cls = GradientBoostingClassifier(n_estimators=n_estimator)
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

c_report = metrics.classification_report(y_test, y_pred)
print('Classification report:\n', c_report)

print("confusion matrix:")
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cf_matrix)
ax = sns.heatmap(cf_matrix, annot=True)
plt.show()
