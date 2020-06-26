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
DATASETDIR = os.path.expanduser('~/data/UCI')
df = pd.read_csv(os.path.join(DATASETDIR, 'data.csv'))

# Set seizure activity label with 1 and the others with zero
# Info about initial class labels: https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
# df['y'] = (df['y'] == 1).replace(True,1).replace(False,0)
print('Class distribution:\n', df.y.value_counts())

# Train/test split
train_df, test_df = skl.train_test_split(df,
                                   test_size = 0.2,
                                   stratify = df['y'])

# Associated features
X_train, y_train, X_test, y_test = train_df.drop(['y','Unnamed: 0'], axis=1), train_df['y'], test_df.drop(['y','Unnamed: 0'], axis=1), test_df['y']

# Scaling features
# sc = StandardScaler()
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# unified sets
train = np.concatenate((X_train, y_train[:,None]), axis=1)
test = np.concatenate((X_test, y_test[:,None]), axis=1)

# Save the processed data
pd.DataFrame(train).to_csv(os.path.join(DATASETDIR,'trainmulticlass.csv'))
pd.DataFrame(test).to_csv(os.path.join(DATASETDIR,'testmulticlass.csv'))

###############################
######## Classifier ###########
###############################

# Supervised transformation based on random forests
# Good to know about feature transformation
n_estimator=100
cls = RandomForestClassifier(max_depth=5, n_estimators=n_estimator)
# cls = GradientBoostingClassifier(n_estimators=n_estimator)
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
