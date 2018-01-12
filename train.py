import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict


train_x = pd.read_csv('UCI HAR Dataset/train/X_train.txt', sep='\s+',header=None)
train_y = pd.read_csv('UCI HAR Dataset/train/y_train.txt', header=None)
train_x = train_x.values
train_y = train_y.values
c,r=train_y.shape
train_y = train_y.reshape(c)
print train_x
print train_y

clf = RandomForestClassifier()
clf.fit(train_x, train_y)
from sklearn.externals import joblib
joblib.dump(clf, 'rf.pkl')


clf = RandomForestClassifier()
clf.fit(train_x[:,:15], train_y)
from sklearn.externals import joblib
joblib.dump(clf, 'rf-simple.pkl')


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 200),
              "min_samples_split": sp_randint(2, 200),
              "min_samples_leaf": sp_randint(1, 200),
              "n_estimators": [5,50,100,200,500,10000],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=20)

random_search.fit(train_x, train_y)
report(random_search.cv_results_)

from sklearn.externals import joblib
joblib.dump(random_search, 'rs.pkl')
