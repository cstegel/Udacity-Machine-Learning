#!/usr/bin/python

import sys
import pickle
from pprint import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedShuffleSplit

# classifiers in use
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, \
    RandomForestClassifier, GradientBoostingClassifier
# TODO: look at bagging estimators like http://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator

# Recursive Feature Elimination with Cross Validation
from sklearn.feature_selection import RFECV # TODO: use this


TEST_CLASSIFIERS = {
    GaussianNB: {},
    MultinomialNB: {},
    BernoulliNB: {},
    SVC: {},
    NuSVC: {'nu': 0.99},
    SGDClassifier: {},
    KNeighborsClassifier: {},
    RadiusNeighborsClassifier: {},
    DecisionTreeClassifier: {},
    AdaBoostClassifier: {},
    RandomForestClassifier: {},
    GradientBoostingClassifier: {},
}

available_features = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'email_address',
    'exercised_stock_options',
    'expenses',
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'loan_advances',
    'long_term_incentive',
    'other',
    'poi',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'shared_receipt_with_poi',
    'to_messages',
    'total_payments',
    'total_stock_value'
]

# method adapted from tester.py that the course instructor/evaluator will use
def get_classifier_scores(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        
        try:
            predictions = clf.predict(features_test)
        except Exception as e:
            return {'classifier': clf, 'exception': e}
            
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            else:
                true_positives += 1
    stats = {'classifier': clf}
    try:
        stats['total_predictions'] = true_negatives + false_negatives + false_positives + true_positives
        stats['accuracy'] = 1.0*(true_positives + true_negatives)/stats['total_predictions']
        stats['precision'] = 1.0*true_positives/(true_positives+false_positives)
        stats['recall'] = 1.0*true_positives/(true_positives+false_negatives)
        stats['f1'] = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        stats['f2'] = (1+2.0*2.0) * stats['precision']*stats['recall'] \
                      / (4*stats['precision'] + stats['recall'])
    except:
        print "Got a divide by zero when trying out:", clf
        for stat in ['true_negatives', 'false_negatives', 'true_positives', 'false_positives']:
            print('%s: %s' % (stat, eval(stat)))
    return stats
    

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"


def main():
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi','salary'] # You will need to use more features

    ### Load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

    ### Task 2: Remove outliers
    # pprint(data_dict)
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    # pprint(my_dataset[my_dataset.keys()[0]])

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    
    # pprint(data)
    labels, features = targetFeatureSplit(data)
    # pprint(labels)
    pprint(features)

    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    # calculate the scores of each classifier
    clf_scores = []
    for clf, kwargs in TEST_CLASSIFIERS.iteritems():
        print('Trying classifier: %s' % str(clf))
        clf_scores.append(get_classifier_scores(clf(**kwargs), my_dataset, features_list))

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script.
    ### Because of the small size of the dataset, the script uses stratified
    ### shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    pprint(clf_scores)
    test_classifier(clf, my_dataset, features_list)

    ### Dump your classifier, dataset, and features_list so 
    ### anyone can run/check your results.

    dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ == '__main__':
    main()

