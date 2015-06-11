#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### the words (features) and authors (labels), already largely processed
### these files should have been created from the previous (Lesson 10) mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (remainder go into training)
### feature matrices changed to dense representations for compatibility with classifier
### functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)



from sklearn.metrics import accuracy_score
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)
print "Accuracy found on the test: {:.4f}".format(acc)

from sklearn.metrics import accuracy_score
pred = clf.predict(features_train)
acc = accuracy_score(labels_train, pred)
print "Accuracy found on the train: {:.4f}".format(acc)


f_max = 0
i_max = 0
i_num = 0
for i,n in enumerate(clf.feature_importances_):
	if n>=0.2: i_num+=1
	if n>f_max:
		f_max = n
		i_max = i

print "The max value was {}, in the position {}".format(f_max, i_max)

print "The most powerful word: {}".format(vectorizer.get_feature_names()[i_max])

print "number of festures with importance >= 0.2: {}".format(i_num)



