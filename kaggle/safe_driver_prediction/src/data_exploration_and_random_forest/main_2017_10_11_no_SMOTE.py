#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:01:27 2017

@author: sling
"""
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew # the target will be log-transformed and so will be some features

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from time import time
import warnings
warnings.filterwarnings('ignore')

##--------Special Libraries
from imblearn.over_sampling import SMOTE, ADASYN

##---------Local Function Files -------------------------------------
from load_data import load_data
from parse_feature_names import parse_feature_names
from impute_and_scale import impute_and_scale
from compute_feature_importances_RF import compute_feature_importances_RF
from calculate_confusion_matrix import calculate_confusion_matrix
from filter_for_top_features import filter_for_top_features
#-----------------------------

# Read the data from the data file
X_all, y_train = load_data("./data/train.csv", "./data/test.csv") 
X_train = X_all[:y_train.shape[0]]
X_test = X_all[y_train.shape[0]:]

_, X_train, _, y_train= train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_all = pd.concat([X_train, X_test])

# Also all NaN's are coded by value -1
X_all = X_all.replace(-1, np.NaN)
print( 'pct of nans in each feature:\n', 100*( X_all.isnull().sum()/X_all.shape[0]) )
## shows  ps_car_03_cat     69.094264, ps_car_05_cat     44.818377
## Remove them from data since so much of data is missing
X_all.drop(['ps_car_03_cat', 'ps_car_05_cat'], axis = 1)

#from collections import Counter
#print( 'Data types of features = ', Counter(X_all.dtypes.values) )
#
## Need to change the dtype of all variables whose names end in _cat
cat_features, bin_features, num_features = parse_feature_names(X_all)  # We are leaving binary features as 0/1
X_all = impute_and_scale(X_all, cat_features, bin_features, num_features, print_time = True)

## TODO
## Find important features
### Let us do PCA on the numerical features on X_all
# from sklearn.decomposition import PCA
# pca = PCA()
# pca.fit(X_all[num_features])
# print(pca.explained_variance_ratio_)
# [ 0.0992199   0.06623136  0.05307663  0.04549893  0.03948505  0.03866309
#   0.03860905  0.03859906  0.03857609  0.03852203  0.03847681  0.03845833
#   0.03843421  0.03841262  0.03840049  0.03836429  0.03834001  0.03832603
#   0.03827192  0.03674315  0.03279999  0.02932834  0.0280729   0.01650862
#   0.00816463  0.00641648]
# print(pca.singular_values_)
# [ 1959.25793529  1600.75198083  1432.99242835  1326.76136491  1235.97176775
#   1223.03957318  1222.18456982  1222.02635812  1221.66272489  1220.80643614
#   1220.08969283  1219.79665875  1219.41414979  1219.07153463  1218.87907796
#   1218.30442721  1217.91887057  1217.69670289  1216.83688241  1192.28587692
#   1126.49452358  1065.21172015  1042.16351717   799.18547722   562.03124506
#    498.24219392]
## As you can tell nothing can be discarded

### I was going to study correlations among the numerical features - but I will go away from linear analysis and focus on tree-based analysis

## Use RandomForest
# Need to separate out the training set from test set so we can add synthetic data to the training set

print('RandomForest on Original Training Data:')
X_train = X_all[:y_train.shape[0]]
X_test = X_all[y_train.shape[0]:]
# create validation set from within the training set
X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
#train and validate
importances, std, y_pred_tmp,  exec_time = compute_feature_importances_RF(X_train_tmp, y_train_tmp, X_test_tmp)  
# test
confusion_matrix= calculate_confusion_matrix(np.ravel( y_test_tmp), np.ravel(y_pred_tmp)) 
true_neg, false_pos, false_neg, true_pos = confusion_matrix[0,0], confusion_matrix[0,1], confusion_matrix[1,0], confusion_matrix[1,1]
print(confusion_matrix) 
print(classification_report(y_test_tmp, y_pred_tmp))
#select features
top_features = filter_for_top_features(X_train.columns, importances, threshold_imps = 0.95)  


## Train
X_all = X_all[top_features]
X_train = X_all[:y_train.shape[0]]
X_test = X_all[y_train.shape[0]:]

tic = time()

from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

# Train and validate
X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

n_estimators_list = [25, 50, 75, 100, 125, 150]
max_depth=8
min_samples_leaf=4
max_features=0.2
n_jobs=-1
random_state=0 
verbose = True
for n_estimators in n_estimators_list:
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=n_jobs, random_state=random_state, verbose = verbose)
    clf.fit(X_train_tmp, y_train_tmp)
    y_pred_tmp = clf.predict(X_test_tmp)
    y_pred_proba_tmp = clf.predict_proba(X_test_tmp)
    confusion_matrix= calculate_confusion_matrix(np.ravel( y_test_tmp), np.ravel(y_pred_tmp))
    
    precision = precision_score(np.ravel( y_test_tmp), np.ravel(y_pred_tmp))
    recall = recall_score(np.ravel( y_test_tmp), np.ravel(y_pred_tmp))
    
    print('Results for : n_estimators = ', n_estimators , '\n')
    print('precision = ', precision, 'recall = ', recall)
    print(confusion_matrix)
    print(y_pred_proba_tmp[:10,1])
toc = time()
print('time taken for 6 runs = ', toc-tic)






