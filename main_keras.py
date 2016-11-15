import numpy as np

np.random.seed(1337)  # for reproducibility

from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from readFile import readDataSet


batch_size = 128
nb_classes = 5
nb_epoch = 3

# from loadDataset import *

data, nrows, ncols = readDataSet("dataset-har-PUC-Rio-ugulino.csv")
# X_train = data[0:50000,0:17]
# y_train = data[0:50000,17]
# X_test = data[60000:70000,0:17]
# y_test = data[60000:70000,17]

X_test = X_train = data[:,0:17]
y_test = y_train = data[:,17]

print X_train, X_test
print y_train, y_test
# (X_train,y_train, X_test,y_test) = loadMNIST();
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# # convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(output_dim=500, input_dim=17, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=250,input_dim=500,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=5,input_dim=250,activation='softmax'))

print model.summary()

sgd = SGD(lr=0.7)
# rMSprop = RMSprop(lr=0.3)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch, 
					verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
y_pred = model.predict_classes(X_test,verbose=0)
print y_pred
print y_test

model.save("Keras_SGD_lR_0_7.h5")

from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import *
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cm,[0,1,2,3,4,5,6,7,8,9],normalize=False,title="Confusion Matrix without normalization with (LR = 0.7)")
plt.savefig("CM_PLOT_Keras_SGD_lR_0_7.png")


