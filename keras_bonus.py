import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Input
# from keras.layers import Dense, Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt


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

inputs = Input(shape=(784,))
encoded = Dense(output_dim=100,activation='sigmoid')(inputs)

decoded  = Dense(784,activation = 'sigmoid')(encoded)
autoencoder = Model(input=inputs,output=decoded)

encoder = Model(input=inputs,output = encoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X_train, X_train,nb_epoch=50,batch_size=256,validation_split=0.1)

autoencoder.save("bonus_autoEncoderTrainedmodel.h5")
encoded_Imgs = encoder.predict(X_train)
encoded_TestImgs = encoder.predict(X_test)

weights = autoencoder.layers[1].get_weights()[0]
for i in range(weights.shape[1]):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(weights[:, i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.clf()

print "encoded_Imgs:", encoded_Imgs
print "encoded TestImgs:",encoded_TestImgs
model = Sequential()
model.add(Dense(output_dim=50, input_dim=100, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=10,input_dim=50,activation='softmax'))

from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import *

lr_ = [0.1,0.3,0.5,0.7,0.9]
# lr_ =[0.1]
MCE = []
for i in lr_:
	print "\nlearning rate: ",i
	sgd = SGD(lr=i)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	history = model.fit(encoded_Imgs, Y_train,batch_size=batch_size, nb_epoch=nb_epoch, 
						verbose=1, validation_split=0.1)
	score = model.evaluate(encoded_TestImgs, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	print('MisClassification Error: ',1-score[1])
	y_pred = model.predict_classes(encoded_TestImgs,verbose=0)
	print y_pred
	print y_test
	MCE.append(1-score[1])
	cm = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
	plt.clf()
	plt.figure()
	plot_confusion_matrix(cm,[0,1,2,3,4,5,6,7,8,9],normalize=False,title="Confusion Matrix without normalization with (LR = "+str(i)+")")
	plt.savefig("CM_PLOT_Keras_autoencoder_"+str(int(i*10))+".png")
	print "   "


print lr_
print MCE
plt.clf()
plt.plot(lr_,MCE,linestyle='-')
plt.xlabel("learning rate")
plt.ylabel("Misclassification Error")
plt.title("Misclassification vs learning rate")
plt.savefig("Bonusplot.png")