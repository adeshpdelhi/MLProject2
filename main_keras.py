import numpy as np

np.random.seed(1337)  # for reproducibility

from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from readFile import readDataSet
from sklearn import preprocessing

from sklearn.feature_selection import VarianceThreshold


batch_size = 256
nb_classes = 5
nb_epoch = 10

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

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print Y_train[0:10]
model = Sequential()
model.add(Dense(output_dim=250, input_dim=18, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=500,input_dim=250,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=5,input_dim=500,activation='softmax'))

print model.summary()
lr = 0.01
sgd = SGD(lr=lr)
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

model.save("Keras_SGD_lR_"+str(lr)+".h5")

from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import *
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cm,[0,1,2,3,4],normalize=False,title="Confusion Matrix without normalization with (LR = "+str(lr)+")")
plt.savefig("CM_PLOT_Keras_SGD_lR_"+str(lr)+".png")


from sklearn.preprocessing import LabelBinarizer
import generateROC

y_score = model.predict(X_test)
print "y_score: ", y_score
plt.clf()
fpr,tpr,thdAddr = generateROC.generate_roc(y_score,Y_test,nROCpts =1000 ,plotROC='false',title="ROC PLOT SGD: lr= "+str(lr))
plt.savefig("ROC_PLOT_Keras_SGD_lR_"+str(lr)+".png")
# print "fpr: ",fpr,"tpr: ",tpr,"thdaddr: ",thdAddr
# return fpr, tpr, thdAddr