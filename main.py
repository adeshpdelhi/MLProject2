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

mlp = MLPClassifier(hidden_layer_sizes=(500, 250),activation='tanh', solver='sgd', 
	alpha=1e-4,tol=1e-3,learning_rate ="constant", learning_rate_init = 0.001, max_iter=20,  verbose=True)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

print "classes:" ,mlp.classes_
print "loss: ",mlp.loss_
print "nlayers: ",mlp.n_layers_
print "noutputs: ",mlp.n_outputs_
print "activation function: ", mlp.out_activation_

print mlp.predict(X_test)
joblib.dump(mlp,"FFNN_sgd_0_001_normalized.pkl")
