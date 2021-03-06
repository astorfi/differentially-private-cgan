## Dataset https://www.kaggle.com/sulianova/cardiovascular-disease-dataset/data?select=cardio_train.csv

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
DATASETDIR = os.path.expanduser('~/data/PTB')
dfabnormal = pd.read_csv(os.path.join(DATASETDIR, 'ptbdb_abnormal.csv'), sep=',').to_numpy()
dfnormal = pd.read_csv(os.path.join(DATASETDIR, 'ptbdb_normal.csv'), sep=',').to_numpy()
df = np.concatenate((dfnormal,dfabnormal), axis=0)

# Set seizure activity label with 1 and the others with zero
# Info about initial class labels: https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
# df['y'] = (df['y'] == 1).replace(True,1).replace(False,0)

# Train/test split
train_df, test_df = skl.train_test_split(df,
                                   test_size = 0.2,
                                   stratify = df[:,-1])

# Report positive labels distribution
print('Positive label percentage in train set', train_df[:,-1].sum()/len(train_df))
print('Positive label percentage in test set', test_df[:,-1].sum()/len(test_df))

# Associated features
X_train, y_train, X_test, y_test = train_df[:,:-1], train_df[:,-1], test_df[:,:-1], test_df[:,-1]

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
print(metrics.confusion_matrix(y_test, y_pred))

# Predic probs
y_pred = cls.predict_proba(X_test)[:, 1]

# ROC
fpr_rf_lm, tpr_rf_lm, _ = metrics.roc_curve(y_test, y_pred)
print('AUROC: ', metrics.auc(fpr_rf_lm,tpr_rf_lm))

# PR
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
AUPRC = metrics.auc(recall, precision)
print('AP: ', metrics.average_precision_score(y_test, y_pred))
print('Area under the precision recall curve: ', AUPRC)
