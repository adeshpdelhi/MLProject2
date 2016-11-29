from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC,LinearSVC
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
import generateROC
import matplotlib.pyplot as plt

import numpy as np

from readFile import readDataSet

data, nrows, ncols = readDataSet("DATASET_WITH_ENCODED_CLASS_NAMES.csv")

X_train = data[:150000, 0:ncols-1]
y_train = data[:150000,ncols-1]
X_test =  data[150000:, 0:ncols-1]
y_test = data[150000:, ncols-1]

def create_subset(size,X,y):
    subX, subY = (np.empty((0,len(X[0]))), np.array([]))
    for x in np.unique(y):
        subset_y,subset_x = (y[y == x],X[y == x])
        subX = np.append(subX, subset_x[:size],axis = 0)
        subY = np.append(subY, subset_y[:size])
    return  subX,subY

def multiClassify(X_train, y_train,X_test,y_test,model):
    Cs = {'estimator__kernel': [model], 'estimator__C': [0.001,0.01,0.1 ,1, 10,100]}        
    if(model == 'rbf'):
        Cs = {'estimator__kernel': [model], 'estimator__C': [0.001,1, 10,], 'estimator__gamma':[1e-8,1e-7, 1e-6,]}         
    clf = GridSearchCV(OneVsRestClassifier(SVC(verbose=True,max_iter=1000),n_jobs=3), Cs, cv=10,verbose=True)
    clf.fit(X_train, y_train)
    print "Best parameter set found:"
    print clf.best_params_
    print "best score:", clf.best_score_
    print "best Estimator: ", clf.best_estimator_
    # print "clf.predict: ", clf.predict(X)
    print "score: ", clf.score(X_test,y_test)
    y_pred = clf.predict(X_test)
    print "actual classes: " ,y_test
    print "predicted classes: ",y_pred
    print "MSE: ", np.sum((y_test- y_pred)**2)/len(y_test)

    from sklearn.metrics import confusion_matrix
    from plot_confusion_matrix import *
    cm = confusion_matrix(y_test, y_pred)
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

    return clf

# # clf = SVC(kernel='linear',C=0.01,verbose = True)
# # One vs rest classifier , LINEAR CLASSFIER, sample of size 2000 per class = 0.01 , accuracy = 82.61
X_train,y_train = create_subset(2000,X_train,y_train)
X_test1 , y_test1 = create_subset(200,X_test,y_test)
print X_train, len(X_train)
print y_train,len(y_train)
kernel = 'linear'
C = 0.0001    
gamma = 1e-10
clf = OneVsRestClassifier(SVC(kernel=kernel,C=C,verbose=True, gamma = gamma),n_jobs = 3)
clf.fit(X_train,y_train)
print clf.predict(X_test1)
print y_test1
print clf.score(X_test1,y_test1)
from sklearn.metrics import f1_score
joblib.dump(clf, "./SVC/multi"+kernel+"_"+str(C)+".pkl")
y_pred = clf.predict(X_test)
print y_pred
print f1_score(y_test, y_pred, average='macro')  

print "kernel: ",kernel
print "C: ",C 
print clf.score(X_test,y_test)

from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import *
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plt.clf()
plot_confusion_matrix(cm,[0,1,2,3,4],normalize=False,title="Confusion Matrix with " +kernel+"; (C = "+str(C)+")")
plt.savefig("./SVC/CM_PLOT_SVM_"+kernel+"_C:"+str(C)+".png")


from sklearn.preprocessing import LabelBinarizer
import generateROC

from keras.utils import np_utils
Y_test = np_utils.to_categorical(y_test, 5)

y_score = clf.decision_function(X_test)
print "y_score: ", y_score
plt.clf()
fpr,tpr,thdAddr = generateROC.generate_roc(y_score,Y_test,nROCpts =1000 ,plotROC='false',title="ROC PLOT :" + kernel+ "; (C: "+str(C)+")")
plt.savefig("./SVC/ROC_PLOT_SVM_"+kernel+"_C:"+str(C)+".png")



#2)
# print "Running linear mutli class classification model"
# clf = multiClassify(X_train,y_train,test_X,test_y,'linear')
# clf = OneVsRestClassifier(SVC(kernel='linear',C=clf.best_params_['estimator__C'],verbose=True),n_jobs = 2)
# clf.fit(X,y)
# y_pred = clf.predict(test_X)
# print "actual Labels: ", test_y
# print "predicted Labels: ", y_pred
# print "MSE: ", np.sum((test_y- y_pred)**2)/len(test_y)
# print "score: ", clf.score(test_X,test_y)
# # print "estimators: ", clf.estimators_
# joblib.dump(clf, "./SVC/multiLinear.pkl")
# for i in range (len(clf.estimators_)):  
#     joblib.dump(clf.estimators_[i], "./SVC/multi"+str(i)+".pkl")

# #3)
# print "running rbf model multi classify"
# X_train,y_train = create_subset(500,X,y)
# X_test,y_test = create_subset(250,test_X,test_y)
# clf = multiClassify(X_train,y_train,test_X,test_y,'rbf')
# print "classifying on 20000 data set :)"
# clf = OneVsRestClassifier(SVC(kernel='rbf',C=clf.best_params_['estimator__C'],gamma=clf.best_params_['estimator__gamma'],verbose=True),n_jobs = 3)
# clf.fit(X,y)
# y_pred = clf.predict(test_X)
# print "actual labels: ", test_y
# print "predicted labels: ", y_pred
# print "MSE: ", np.sum((test_y- y_pred)**2)/len(test_y)
# print "score: ", clf.score(test_X,test_y)
# # print clf.estimators_
# joblib.dump(clf, "./../Models/multirbf.pkl")
# for i in range (len(clf.estimators_)):  
#     joblib.dump(clf.estimators_[i], "./../Models/rbf"+str(i)+".pkl")
