#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
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
	from sklearn import tree


	###slice down the training data 



	### create classifier
	clf = tree.DecisionTreeClassifier(min_samples_split=40)
	

	### fit the classifier on the training features and labels
	print "start training"
	t0 = time()
	clf.fit(features_train, labels_train)

	print "training time:", round(time()-t0, 3), "s"

	### use the trained classifier to predict labels for the test features
	t0 = time()

	### calculate and return the accuracy on the test data
	### this is slightly different than the example, 
	### where we just print the accuracy
	### you might need to import an sklearn module
	#accuracy = clf.score(features_test, labels_test)



	#another way
	from sklearn.metrics import accuracy_score
	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)
	print "Accuracy found : {:.4f}".format(acc)


	print "predicting time:", round(time()-t0, 3), "s"

	# return accuracy





 



if __name__ == "__main__":
	#calculate the accuracy
	print 'number of features: {}'.format( len(features_train[0]))
	x = measure_accuracy(features_train, labels_train, features_test, labels_test)






#########################################################


