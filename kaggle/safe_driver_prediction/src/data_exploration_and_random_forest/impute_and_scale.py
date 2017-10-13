#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:25:33 2017

@author: sling
"""

def impute_and_scale(X, cat_features, bin_features, num_features, print_time = True):
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  
  from time import time
  tic = time()
  X_cp = X.copy()
  # change data types of cat to object
  for cat in cat_features:
    X_cp[cat] = X_cp[cat].astype(object)
  
#  from collections import Counter
#  print( 'Data types of features = ', Counter(X_cp.dtypes.values) )
  
  # impute, center and scale num_features
  
  X_cp[num_features] = X_cp[num_features].fillna(X_cp[num_features].mean())
  
  ss = StandardScaler()
  X_cp[num_features] = ss.fit_transform(X_cp[num_features])
  
  # imputing, making sure 0/1 for binary features  - tested to find that all are 0 or 1
  # for f in bin_features:
  #     a0,a1 = np.bincount(X_all[f])
  #     b0 = (X_all[f] == 0).sum()
  #     b1 = (X_all[f] == 1).sum()
  #     print(a0, a1, a0+a1, b0, b1, b0+b1)
  
  # replacing categorical features by dummies
  X_cp = pd.get_dummies(X_cp)
  toc = time()
  if print_time:
    print('time to fix nan and scale the num_feartures = ', toc-tic, 'sec.')
  return X_cp