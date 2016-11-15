from mnist import MNIST
import numpy as np

def create_subset(size,X,y):
    subX, subY = (np.empty((0,28*28)), np.array([]))
    for x in np.unique(y):
        subset_y,subset_x = (y[y == x],X[y == x])
        subX = np.append(subX, subset_x[:size],axis = 0)
        subY = np.append(subY, subset_y[:size])
    return  subX,subY

def loadMNIST():
	mndata = MNIST('./DATAFILES')
	train_images, train_labels =  mndata.load_training()
	test_images, test_labels = mndata.load_testing()

	X_train, y_train = (np.array(train_images),np.array(train_labels))
	X_test, y_test = (np.array(test_images),np.array(test_labels))
	return (X_train,y_train, X_test,y_test)
# print len(train_images), len(test_images)
