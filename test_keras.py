import numpy as np

np.random.seed(1337)  # for reproducibility

from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import load_model

batch_size = 128
nb_classes = 10
nb_epoch = 10

from loadDataset import *

(X_train,y_train, X_test,y_test) = loadMNIST();
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# # convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = load_model("./Keras SGD/Models/Keras_SGD_lR_0_0001.h5")
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
y_pred = model.predict_classes(X_test,verbose=0)
print y_pred
print y_test

from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import *
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cm,[0,1,2,3,4,5,6,7,8,9],normalize=False,title="Confusion Matrix without normalization with (LR = 0.0001)")
plt.show()

