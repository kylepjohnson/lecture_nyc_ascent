"""Script to demo scikit for tweet popular/unpopular classification.

Author: Cesar Koirala (2/23/2016)
"""

import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import svm, tree
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


def csv_to_dict(csv_filename):
    # Let's say, We are intersted in only count features
    count_features = ['_char_count', '_hashtag_count', '_word_count', '_url_count']
    with open(csv_filename) as f:
        features = [({k: int(v) for k, v in row.items() if k in count_features}, row['_popular'])
                    for row in csv.DictReader(f, skipinitialspace=True)]
        X = [f[0] for f in features]
        Y = [f[1] for f in features]
    return (X, Y)


def train(csv_filename):
    features = csv_to_dict(csv_filename)

    vec = DictVectorizer()
    data = features[0]
    target = features[1]
    X = vec.fit_transform(data).toarray()  # change to numpy array
    Y = np.array(target)  # change to numpy array

    '''
    -In case we need to know the features
    '''
    feature_names = vec.get_feature_names()
    print feature_names

    '''
    -Dividing the data into train and test
    -random_state is pseudo-random number generator state used for
     random sampling
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

    '''
    -PREPOCESSING 
    -Here, scaled data has zero mean and unit varience
    -We save the scaler to later use with testing/prediction data
    '''
    scaler = preprocessing.StandardScaler().fit(X_train)
    joblib.dump(scaler, 'scaler.pkl')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    '''
    -This is where we define the models
    -Here, I use SVM and Decision tree with pre-defined parameters
    -We can learn these parameters given our data
    '''
    clf0 = svm.LinearSVC(C=100.)
    clf1 = tree.DecisionTreeClassifier()

    clf0.fit(X_train_scaled, Y_train)
    clf1.fit(X_train_scaled, Y_train)

    joblib.dump(clf0, 'svc.pkl')
    joblib.dump(clf1, 'tree.pkl')

    Y_prediction_svc = clf0.predict(X_test_scaled)
    print 'svc_predictions ', Y_prediction_svc
    Y_prediction_tree = clf1.predict(X_test_scaled)
    print 'tree_predictions ', Y_prediction_tree
    expected = Y_test
    print 'actual_values   ', expected

    '''
    Classifiation metrics
    (Case 1): SVMs
    '''
    print
    print '----Linear SVC_report--------------------------'
    print classification_report(expected, Y_prediction_svc)

    '''
    Classification metrics
    (case 2): Decision tree
    '''
    print
    print '----Tree_report--------------------------------'
    print classification_report(expected, Y_prediction_tree)


if __name__ == "__main__":
    train("feature_tables/test.csv")
