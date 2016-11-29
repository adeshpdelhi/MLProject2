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
from sklearn.preprocessing import StandardScaler,Normalizer

def fit(X_train, Y_train, X_test,Y_test, input_dim, output_dim, solver = 'sgd', lr = 0.01, nlayers = 2, 
	neuronsPerLayer = [250,500], batch_size = 256, nb_epoch = 10, verbose = 1, validation_split = 0.1, 
	metrics=['accuracy'],loss = 'categorical_crossentropy'):

	if(nlayers != len(neuronsPerLayer)): 
		print "len(neuronsPerLayer] != nlayers"
		return -1
	print "solver: ",solver
	print "learning Rate: ", lr
	print "nlayers: ",nlayers
	print "neurons per layer: ", neuronsPerLayer
	print "epochs: ",nb_epoch
	model = Sequential()
	model.add(Dense(output_dim=neuronsPerLayer[0], input_dim=input_dim, activation='tanh'))
	model.add(Dropout(0.2))

	for i in range (1, nlayers):
		model.add(Dense(output_dim=neuronsPerLayer[i],input_dim=neuronsPerLayer[i-1],activation='tanh'))
		model.add(Dropout(0.2))

	model.add(Dense(output_dim=output_dim,input_dim=neuronsPerLayer[nlayers-1],activation='softmax'))
	print model.summary()
	if(solver == 'sgd'):
		optimizer = SGD(lr=lr)
	elif(solver == 'RMSprop'):
		optimizer = RMSprop(lr=lr)
	else:
		optimizer = SGD(lr=lr)

	model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

	history = model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch, 
						verbose=verbose, validation_split=validation_split)
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	y_pred = model.predict_classes(X_test,verbose=0)
	print y_pred
	print y_test
	from sklearn.metrics import f1_score
	print f1_score(y_test, y_pred, average='macro')  

	model.save("Keras_"+solver+"_hiddenlayers:"+str(nlayers)+" "+str(neuronsPerLayer)+"_lR_"+str(lr)+".h5")

	from sklearn.metrics import confusion_matrix
	from plot_confusion_matrix import *
	cm = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
	plt.clf()
	plot_confusion_matrix(cm,[0,1,2,3,4],normalize=False,title="Confusion Matrix with " +solver+", hiddenlayers:"+str(nlayers)+" "+str(neuronsPerLayer)+";(LR = "+str(lr)+")")
	plt.savefig("CM_PLOT_Keras_"+solver+"_hiddenlayers:"+str(nlayers)+" "+str(neuronsPerLayer)+"_lR_"+str(lr)+".png")


	from sklearn.preprocessing import LabelBinarizer
	import generateROC

	y_score = model.predict(X_test)
	print "y_score: ", y_score
	plt.clf()
	fpr,tpr,thdAddr = generateROC.generate_roc(y_score,Y_test,nROCpts =1000 ,plotROC='false',title="ROC PLOT SGD: lr= "+str(lr))
	plt.savefig("ROC_PLOT_Keras_"+solver+"_hiddenlayers:"+str(nlayers)+" "+str(neuronsPerLayer)+"_lR_"+str(lr)+".png")
	return score[1]

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

# print "Standardized data :)"
# stdscl = StandardScaler()

# X_train = stdscl.fit_transform(X_train,y_train)
# X_test = stdscl.fit_transform(X_test,y_test)
# print "Normalized :)"
# X_train = Normalizer().fit_transform(X_train,y_train)
# X_test = Normalizer().fit_transform(X_test,y_test)
# print X_train[0:10]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print Y_train[0:10]
input_dim = ncols-1
output_dim = nb_classes
lrs = [0.0001,0.001,0.05,0.02,0.01,0.5,0.1]
bestAcc = 0
nlayers = 0
l_r = 0
nlyr = [2,3,4]
neurons = [50,100,200,400]
nFits = 0
acc = fit(X_train,Y_train,X_test,Y_test,input_dim,output_dim,lr=0.01,nlayers = 2, 
				neuronsPerLayer=[1000,1000])

# for i in range(0,len(nlyr)):
# 	for j in range(0,len(neurons)-nlyr[i]+1):
# 		nNrns = neurons[j:j+nlyr[i]]
# 		for k in range (0,len(lrs)):
# 			nFits = nFits + 1
# 			acc = fit(X_train,Y_train,X_test,Y_test,input_dim,output_dim,lr=lrs[k],nlayers = nlyr[i], 
# 				neuronsPerLayer=nNrns)
# 			if(bestAcc < acc):
# 				bestAcc = acc
# 				nlayers = nlyr[i]
# 				l_r = lrs[k]
# 				neuronsPerLayer = nNrns 

# print "best Accuracy :",bestAcc
# print "nlayers: ",nlayers
# print "neuronsPerLayer: ",neuronsPerLayer
# print "learning_rate: ", l_r
# # print "fpr: ",fpr,"tpr: ",tpr,"thdaddr: ",thdAddr
# # return fpr, tpr, thdAddr