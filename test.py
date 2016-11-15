from sklearn.externals import joblib
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

import numpy as np

from loadDataset import *

(X_train,y_train, X_test,y_test) = loadMNIST();
print len(X_train), len(X_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

mlp = joblib.load('./Models/FFNN_0.1.pkl')
print mlp
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

print "classes:" ,mlp.classes_
print "loss: ",mlp.loss_
print "nlayers: ",mlp.n_layers_
print "noutputs: ",mlp.n_outputs_
print "activation function: ", mlp.out_activation_

y_pred =  mlp.predict(X_test)
print y_pred
print y_test

from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import *
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cm,mlp.classes_,normalize=False,title="Confusion Matrix with (LR = 0.1)")
# plt.savefig("CM_PLOT_sgd_lR_0_1_normalized.png")
plt.show()

