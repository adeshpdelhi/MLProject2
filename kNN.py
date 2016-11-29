import numpy as np

np.random.seed(1337)  # for reproducibility

from sklearn.externals import joblib
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from readFile import readDataSet
from sklearn import preprocessing

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler,Normalizer

data, nrows, ncols = readDataSet("DATASET_WITH_ENCODED_CLASS_NAMES.csv")

X_train = data[:150000, 0:ncols-1]
y_train = data[:150000,ncols-1]
X_test =  data[150000:, 0:ncols-1]
y_test = data[150000:, ncols-1]

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_red = sel.fit_transform(X_train)

print X_red[0:10] , len(X_red[0])
print X_train[0:10], len(X_train[0])
print y_train[0:10]


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = np_utils.to_categorical(y_train, 5)
Y_test = np_utils.to_categorical(y_test, 5)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3,weights = 'distance' ,n_jobs = 3)
print(np.mean(cross_val_score(neigh, X_train, y_train,cv=10,scoring='f1'))) 

neigh.fit(X_train, y_train) 
y_pred = neigh.predict(X_test)
y_score =  neigh.predict_proba(X_test)
print neigh.score(X_test,y_test)


from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import *
cm = confusion_matrix(y_test, y_pred)
plt.clf()
plot_confusion_matrix(cm,[0,1,2,3,4],normalize=False,title="Confusion Matrix; KNN, neighbors = 3")
plt.savefig("CM_PLOT_KNN_neighbors=3.png")


from sklearn.preprocessing import LabelBinarizer
import generateROC
print "y_score: ", y_score
plt.clf()
fpr,tpr,thdAddr = generateROC.generate_roc(y_score,Y_test,nROCpts =1000 ,plotROC='false',title="ROC PLOT KNN; neighbors = 3")
plt.savefig("ROC_PLOT_KNN_neighbors=3.png")

