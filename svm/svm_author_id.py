#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#Create the method to compute the 
def measure_accuracy(features_train, labels_train, features_test, labels_test):
	""" compute the accuracy of your Naive Bayes classifier """
	### import the sklearn module for GaussianNB
	from sklearn.svm import SVC


	###slice down the training data 
	features_train_small = features_train[:len(features_train)/100] 
	labels_train_small = labels_train[:len(labels_train)/100]	

	### create classifier
	clf = SVC(kernel="linear")
	l_model=[]
	l_c=[10.0, 100., 1000., 10000.]
	for f_c in l_c:
		l_model.append(SVC(kernel="rbf", C = f_c))

	### fit the classifier on the training features and labels
	print "calculating for a small sample"
	t0 = time()
	l_clf=[]
	for clf in l_model:
		l_clf.append(clf.fit(features_train_small, labels_train_small))
	
	# print "calculating for the total sample"
	clf_all = SVC(kernel="rbf", C = 10000.)
	clf_all = clf_all.fit(features_train, labels_train)

	print "training time:", round(time()-t0, 3), "s"

	### use the trained classifier to predict labels for the test features
	t0 = time()
	l_pred=[]
	for clf in l_clf:
		l_pred.append(clf.predict(features_test))

	pred_all = clf_all.predict(features_test)

	### calculate and return the accuracy on the test data
	### this is slightly different than the example, 
	### where we just print the accuracy
	### you might need to import an sklearn module
	#accuracy = clf.score(features_test, labels_test)



	#another way
	from sklearn.metrics import accuracy_score
	for pred,c in zip(l_pred, l_c):
		acc = accuracy_score(pred, labels_test)
		print "Accuracy found for C={}: {:.4f}".format(c, acc)
	
	acc = accuracy_score(pred_all, labels_test)
	print "Accuracy found for C={} to all data: {:.4f}".format(10000, acc)
	
	print "prediction to the element {}: {}".format(26,l_pred[-1][10])
	print "prediction to the element {}: {}".format(26,l_pred[-1][26])
	print "prediction to the element {}: {}".format(26,l_pred[-1][50])

	print "summing all results: {}".format(sum(pred_all))

	print "predicting time:", round(time()-t0, 3), "s"

	# return accuracy





 



if __name__ == "__main__":
	#calculate the accuracy
	x = measure_accuracy(features_train, labels_train, features_test, labels_test)


#########################################################


