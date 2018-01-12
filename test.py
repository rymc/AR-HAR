import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict


test_x = pd.read_csv('UCI HAR Dataset/test/X_test.txt', sep='\s+',header=None)
test_y = pd.read_csv('UCI HAR Dataset/test/y_test.txt', header=None)
print test_x
print test_y
test_x = test_x.values
test_y =test_y.values
c,r=test_y.shape
test_y = test_y.reshape(c)

from sklearn.externals import joblib
clf =joblib.load('rf.pkl')
clf_s = joblib.load('rf-simple.pkl')
pred = clf.predict(test_x)
pred_s = clf_s.predict(test_x[:,:15])

from sklearn import metrics

print metrics.f1_score(test_y, pred, average='macro')

print metrics.confusion_matrix(test_y,pred)


print metrics.f1_score(test_y, pred_s, average='macro')

print metrics.confusion_matrix(test_y,pred_s)


#feat_names = pd.read_csv('UCI HAR Dataset/features.txt', sep='\s+', header=None, index_col=0)
#feat_names = feat_names.to_dict()
#print feat_names
feat_names = {}
with open('UCI HAR Dataset/features.txt') as f:
    for line in f:
        line = line.rstrip()
        line = line.split(" ")
        feat_names[int(line[0])] = line[1]
        
from pprint import pprint
pprint(feat_names)

print "whats the most important features?"
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print indices
print max(indices)
print min(indices)


for f in range(test_x.shape[1]):
#        print "f"
#        print f
#        print "==="
#        print "indices"
#        print indices[f]
      #  if indices[f] == 0:
      #      print "FOUND 0, skipping"
      #      print f
      #      continue
#        print "xxx"
        idx = indices[f]
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        print("%d. %s (%f)" % (f + 1, feat_names[idx+1], importances[idx]))

print feat_names[1]
print feat_names[115]
