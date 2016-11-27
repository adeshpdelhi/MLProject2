from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC,LinearSVC
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt


from mnist import MNIST
import numpy as np
import generateROC
from sklearn.metrics import roc_curve,auc

def create_subset(size,X,y):
    subX, subY = (np.empty((0,28*28)), np.array([]))
    for x in np.unique(y):
        subset_y,subset_x = (y[y == x],X[y == x])
        subX = np.append(subX, subset_x[:size],axis = 0)
        subY = np.append(subY, subset_y[:size])
    return  subX,subY

mndata = MNIST('./../DATAFILES')
train_images, train_labels =  mndata.load_training()
test_images, test_labels = mndata.load_testing()

train_images, train_labels = (np.array(train_images),np.array(train_labels))
test_images, test_labels = (np.array(test_images),np.array(test_labels))

X, y = create_subset(2000,train_images,train_labels)
test_X, test_y = create_subset(500,test_images,test_labels)
binaryX = test_X[(test_y == 3) | (test_y == 8)]
binaryY = test_y[(test_y == 3) | (test_y == 8)]
# clf = joblib.load('./model_linear.pkl')
def load_and_Plot(file,plotROC):
	# ./multirbf.pkl, ./multiLinear.pkl ./model_linear
	if(file == 'model_linear.pkl'):		
		X_test = binaryX
		y_test = binaryY
	else:
		X_test = test_X
		y_test = test_y
	clf = joblib.load('./../Models/'+ file)
	pred = clf.predict(X_test)
	print "actual classes: ", y_test
	print "predicted classes: ", pred,len(pred)
	print "MSE: ", np.sum((pred-y_test )**2)/len(y_test)
	print "score: ", clf.score(X_test,y_test)

	y_score = clf.decision_function(X_test)
	if(file == 'model_linear.pkl'):
		test = (y_test == 8)*1
	else:
		test = np.zeros((10,5000))
		for i in range(10):
		    test[i,:] = (y_test == i)
	print test
	fpr,tpr,thdAddr = generateROC.generate_roc(y_score.transpose(),test,nROCpts =1000 ,plotROC = plotROC)
	print "fpr: ",fpr,"tpr: ",tpr,"thdaddr: ",thdAddr
	return fpr, tpr, thdAddr

def mutliPlot():
	fpr,tpr ,t1= load_and_Plot('multiLinear.pkl','false')
	plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
	plt.plot(fpr[0,:],tpr[0,:], 'b.-',color='red', label='multi Linear')
	fpr ,tpr, t = load_and_Plot('multirbf.pkl','false')
	plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
	plt.plot(fpr[0,:],tpr[0,:], 'b.-',color='blue', label='multi rbf')
	plt.legend(loc="lower right")
	plt.show()
	
# mutliPlot()
# load_and_Plot('multirbf.pkl','true')
# load_and_Plot('multiLinear.pkl','true')
load_and_Plot('model_linear.pkl','true')