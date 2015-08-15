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

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


features_train, features_test, labels_train, labels_test = \
  train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
res = clf.predict(features_test)
recall = recall_score(labels_test, res)
precision = precision_score(labels_test, res)

print('precision: %s' % precision)
print('recall: %s' % recall)

poi_test = [x for x in labels_test if x == True]
predicted_poi_test = [x for x in res if x == True]

true_pos = 0
for i, x in enumerate(res):
  if x == True and labels_test[i] == True:
    true_pos += 1
print('true pos: %s' % true_pos)

print('actual poi: %s, res: %s' % (len(poi_test), len(predicted_poi_test)))
bad_acc = accuracy_score(labels_test, clf.predict(features_test))

print('bad accuracy: %s' % bad_acc)

