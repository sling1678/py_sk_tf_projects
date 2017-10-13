#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:09:19 2017

@author: sling
"""
def select_important_features(X_all, y_train, transform_X = True, choose_RF = True,  threshold_imps  = 0.9, print_time = True, plot_imp = True):
  from sklearn.metrics import precision_recall_fscore_support
  from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
  from sklearn.grid_search import GridSearchCV
  from time import time
  import numpy as np
  import matplotlib.pyplot as plt
  tic = time()
  params = {'n_estimators'  : [10], 'max_depth' : [6], 'min_samples_split' : [2], }
  n_estimators , max_depth , min_samples_split =  10, 6, 2
  if choose_RF:
    clf = RandomForestClassifier( n_jobs=-1, random_state=0,n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split)
  else:
    clf = ExtraTreesClassifier(random_state=0, n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split)
  #gscv = GridSearchCV(forest, param_grid = params, n_jobs = -1, cv=5, scoring='recall')  
  recall_max = 0
  changed = False
  for n_estimators in params['n_estimators' ]:
    for max_depth in params['max_depth' ]:
      for min_samples_split in params['min_samples_split']:
        if not changed:
          best_params = [n_estimators, max_depth, min_samples_split]
        clf.fit(X_all[:y_train.shape[0]], y_train)
        y_pred = clf.predict(X_all[:y_train.shape[0]])
        precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_pred)
        if recall > recall_max:
          recall_max = recall
          best_params = [n_estimators, max_depth, min_samples_split]
          changed = True
  n_estimators, max_depth, min_samples_split = best_params
  clf.fit(X_all[:y_train.shape[0]], y_train)
  y_pred = clf.predict(X_all[:y_train.shape[0]])
  precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_pred)        
        
  importances = clf.feature_importances_
  std = np.std([tree.feature_importances_ for tree in clf.estimators_],  axis=0)
  indices = np.argsort(importances)[::-1]
  toc = time()
   
  if print_time:
    print('time for selecting features is ', toc-tic, 'sec') # 21.6 minutes wow
  
  if plot_imp:
    print("Feature ranking:")
    for f in range(X_all.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_all.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_all.shape[1]), indices)
    plt.xlim([-1, X_all.shape[1]])
    plt.show()
  
  #prepare to return modified X_all
  if transform_X:
    feats = X_all.columns
    idx, total, threshold_imps = 0, 0, threshold_imps 
    # threshold_imp can be adjusted  in a larger loop , but it takes too long in my computer
    while total < threshold_imps:
        total = total + importances[idx]
        idx += 1
    important_feats=feats[indices[:idx]]
    X_all = X_all[important_feats]
  
  return X_all, gscv