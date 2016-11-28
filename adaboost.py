from sklearn.externals import joblib
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

import numpy as np

from readFile import readDataSet

data, nrows, ncols = readDataSet("DATASET_WITH_ENCODED_CLASS_NAMES.csv")

X_train = data[:150000, 0:ncols-1]
y_train = data[:150000,ncols-1]
X_test =  data[150000:, 0:ncols-1]
y_test = data[150000:, ncols-1]

# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# X_red = sel.fit_transform(X_train)

# print X_red[0:10] , len(X_red[0])
print X_train[0:10], len(X_train[0])
print y_train[0:10]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier
print "ADABOOST cLASSIFIER: "
clf = AdaBoostClassifier(n_estimators=100,learning_rate = 0.01)
scores = cross_val_score(clf, X_train, y_train)
print "scores mean: ",scores.mean()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print "original y_train: ",y_train
print "pred train:",y_pred
print "train score:", clf.score(X_train,y_train)

print "orignial y_test:",y_test
print "pred test:",clf.predict(X_test)
print "test score:", clf.score(X_test,y_test)
