#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

## import the sklearn module
from sklearn import tree
from time import time
import numpy as np
from sklearn import cross_validation
from sklearn import datasets


###slice down the training data 

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
	features, labels, test_size=0.3, random_state=42)

### create classifier
clf = tree.DecisionTreeClassifier()


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


