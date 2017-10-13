#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:52:19 2017

@author: sling
"""

def compute_feature_importances_RF(X_train, y_train, X_test, n_estimators=100, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0,  verbose = True):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.ensemble import  RandomForestClassifier
    from time import time
    import numpy as np
     
    tic = time()
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=n_jobs, random_state=random_state, verbose = verbose, bootstrap = True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
   # precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_pred)  
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],  axis=0)
 #   indices = np.argsort(importances)[::-1]
#  
#    feats = X_all.columns
#    idx, total, threshold_imps = 0, 0, threshold_imps 
#   # threshold_imp can be adjusted  in a larger loop , but it takes too long in my computer 
#    while total < threshold_imps:
#        total = total + importances[idx]
#        idx += 1
#    important_feats=feats[indices[:idx]]
    toc = time()
    exec_time = toc-tic
    return importances, std, y_pred,  exec_time
  
  
  