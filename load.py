from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC,LinearSVC
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
import generateROC
import matplotlib.pyplot as plt

import numpy as np

from readFile import readDataSet

data, nrows, ncols = readDataSet("DATASET_WITH_ENCODED_CLASS_NAMES.csv")

X_train = data[:150000, 0:ncols-1]
y_train = data[:150000,ncols-1]
X_test =  data[150000:, 0:ncols-1]
y_test = data[150000:, ncols-1]

clf = joblib.load("./SVC/multilinear_0.001.pkl")
print clf
y_pred = clf.predict(X_test)
print y_pred
print clf.score(X_test,y_test)
from sklearn.metrics import f1_score
print f1_score(y_test, y_pred, average='macro')
from sklearn.model_selection import cross_val_score
print(np.mean(cross_val_score(clf, X_train, y_train,cv=10,scoring='neg_log_loss'))) 


