#!/usr/bin/env python3
#!export PYTHONPATH=~/xgboost/python-package
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:56:14 2017

@author: sling
"""

import sys
sys.path.append('~/xgboost/python-package')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score, make_scorer, roc_auc_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer

#--------Local functions
from parse_feature_names import parse_feature_names
from impute_and_scale import impute_and_scale

#---------- Load Files--------------
from pathlib import Path
from six.moves import cPickle as pickle
import os.path

from save_pickled_data import save_pickled_data

train_pickle = '../../data/train.pickle'
test_pickle = '../../data/test.pickle'

train_pickle_cleaned = '../../data/train_cleaned.pickle'
test_pickle_cleaned = '../../data/test_cleaned.pickle'

if not ( os.path.isfile(train_pickle_cleaned) and os.path.isfile(test_pickle_cleaned) ):
    if not ( os.path.isfile(train_pickle) and os.path.isfile(test_pickle) ):
        print('load files...')
        train_df = pd.read_csv('../../data/train.csv', na_values=-1) #na_values are given to be -1 in the dataset
        test_df = pd.read_csv('../../data/test.csv', na_values=-1)
        #-----------Reduce the size of files
        for col in train_df.select_dtypes(include=['float64']).columns[2:]:
          train_df[col]=train_df[col].astype(np.float32) #cut down space requirement
          test_df[col]=test_df[col].astype(np.float32)
        for col in train_df.select_dtypes(include=['int64']).columns[2:]:
          train_df[col]=train_df[col].astype(np.int8) #cut down space requirement
          test_df[col]=test_df[col].astype(np.int8)

        #-------save as pickled file

        save_pickled_data(train_df, train_pickle)
        save_pickled_data(test_df, test_pickle)
    else:
        with open(train_pickle, 'rb') as f:
          train_df = pickle.load(f)
        with open(test_pickle, 'rb') as f:
          test_df = pickle.load(f)

    # #-----------Check NANs
    # print('Number of NaNs in the entire training data set = ', train_df.isnull().sum().sum())
    # print('Number of NaNs in the entire test data set = ', test_df.isnull().sum().sum())
    cat_features, bin_features, num_features, other_features = parse_feature_names(train_df)

    # print(cat_features)
    # print(num_features)
    # print(other_features)

    feature_types = [num_features, cat_features, bin_features]
    for feature in feature_types:
        # print('Number of NaNs in the train data set = ', train_df[feature].isnull().sum().sum())
        # print('Number of NaNs in the test data set = ', test_df[feature].isnull().sum().sum())

        num_full_data_to_fit = pd.concat([train_df[feature], test_df[feature]])
        if feature == num_features:
            imp = Imputer(strategy='mean')
            imp.fit(num_full_data_to_fit)
            train_df[feature] = imp.transform(train_df[feature])
            test_df[feature] = imp.transform(test_df[feature])

        else:
            imp = Imputer(strategy='most_frequent')
            imp.fit(num_full_data_to_fit)
            train_df[feature] = imp.transform(train_df[feature])
            test_df[feature] = imp.transform(test_df[feature])
        # print('Number of NaNs in the entire training data set, feature = ', feature, train_df[feature].isnull().sum().sum())
        # print('Number of NaNs in the test data set , feature = ', feature, test_df[feature].isnull().sum().sum())

    num_full_data_to_fit = pd.concat([train_df[num_features], test_df[num_features]])
    ss = StandardScaler()
    ss.fit(num_full_data_to_fit)
    train_df[num_features] = ss.transform(train_df[num_features])
    test_df[num_features] = ss.transform(test_df[num_features])


    cat_full_data_to_fit = pd.concat([train_df[cat_features], test_df[cat_features]])
    mms = MinMaxScaler()
    mms.fit(cat_full_data_to_fit)
    train_df[cat_features] = mms.transform(train_df[cat_features])
    test_df[cat_features] = mms.transform(test_df[cat_features])

    save_pickled_data(train_df, train_pickle_cleaned)
    save_pickled_data(test_df, test_pickle_cleaned)
else:
    with open(train_pickle_cleaned, 'rb') as f:
        train_df = pickle.load(f)
    with open(test_pickle_cleaned, 'rb') as f:
        test_df = pickle.load(f)

#print(train_df.head())
#----------Check correlations of X_data with y_data in the train

cat_features, bin_features, num_features, other_features = parse_feature_names(train_df)
corrmat = pd.concat([ train_df[num_features], train_df[other_features]], axis = 1).corr()
cols = corrmat.nlargest(corrmat.shape[0], 'target')['target'] ## will plot all
cols = cols.drop(['target', 'id'])
#cols.plot(kind = 'barh') # col is pd.Series -to plot we use Series.plot
#
# plt.figure()
# cols.plot(kind = 'barh')
# plt.show()

corr_threshold = 0.05*np.abs(cols.max())
cols_with_lowest_corr = []
for idx in cols.index:
    if np.abs(cols.loc[idx])  < corr_threshold:
        cols_with_lowest_corr.append(idx)

#print(np.sort(cols_with_lowest_corr))
#print(corr_threshold)

# For linear learning algorithms we may say that if corr is small then, there is not linear dependence of y
#  on that feature. Low corr does not mean statistical independence even though statistical independence
# implies sero corr, the converse is however not true.

train_df_linear = train_df.drop(cols_with_lowest_corr, axis = 1)
test_df_linear = test_df.drop(cols_with_lowest_corr, axis = 1)

# For non-linear algorithms, such as tree-based algorithms, we can use feature_importance_ to prune features

#print(train_df[cat_features].head(10))

#print(train_df_linear.head(1))
#print(test_df_linear.head(1))

test_id = test_df_linear['id']
y = train_df['target'].values

train_df_linear = train_df_linear.drop(['target', 'id'], axis = 1)
test_df_linear = test_df_linear.drop(['id'], axis = 1)

features = train_df_linear.columns
X = train_df_linear.values  ## convert to ndarray
X_test = test_df_linear.values

# print(X.shape, y.shape)
# print(X_test.shape)
#
# kfold = 5
# skf = StratifiedKFold(n_splits=kfold, random_state=0)
# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     print('kfold: {}  of  {} : '.format(i+1, kfold))
#     X_train, X_valid = X[train_index], X[test_index]
#     y_train, y_valid = y[train_index], y[test_index]
#     print(X_train.shape, X_valid.shape)
#     print(y_train.shape, y_valid.shape)
#
#     logistic = LogisticRegression(penalty='l1', verbose = 1)
#
#     fit_train = logistic.fit(X_train, y_train)
#     y_pred_valid = logistic.predict(X_valid)
#
#     print('AUC = ', roc_auc_score(y_valid, y_pred_valid))


#--- xgb

#--------------------
# custom objective function (similar to auc)
#
# def gini(y, pred):
#     g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
#     g = g[np.lexsort((g[:,2], -1*g[:,1]))]
#     gs = g[:,0].cumsum().sum() / g[:,0].sum()
#     gs -= (len(y) + 1) / 2.
#     return gs / len(y)
#
# def gini_xgb(pred, y):
#     y = y.get_label()
#     return 'gini', gini(y, pred) / gini(y, y)
#
# def gini_lgb(preds, dtrain):
#     y = list(dtrain.get_label())
#     score = gini(y, preds) / gini(y, y)
#     return 'gini', score, True
# #----

def gini_xgb(x,y):
    '''
    gini coefficient was introduced in economics to quantify income inequality in a society.

    :param x: property whose cumsum will be on x-axis, if it is 0, 1, 2, ..., x = [1, 1, 1, ...]
    :param y: property whose cumsum will be on y-axis, if it is 0, 1, 2, ..., x = [1, 1, 1, ...]
    :return: Gini coefficient between 0 and 1
    '''
    y1, y2 = [], []
    sum_x, sum_y = 0, 0
    for i in range(len(x)):
        sum_x += x[i]
        y1.append(sum_x)
    for i in range(len(x)):
        sum_y += y[i]
        y2.append(sum_y)
    for i in range(len(x)):
        y1[i] = y1[i] - x[0]
    for i in range(len(x)):
        y2[i] = y2[i] - y[0]

    # Area under Lorentz curve
    B12, B21 = 0, 0
    for i in range(len(x) - 1):
        B12 += (y1[i + 1] + y1[i]) * (y2[i + 1] - y2[i]) / 2
        B21 += (y2[i + 1] + y2[i]) * (y1[i + 1] - y1[i]) / 2
    R = (y1[-1] - y1[0]) * (y2[-1] - y2[0]) / 2
    A12, A21 = R - B12, R - B21
    if A12 >= 0:
        G = A12 / R
    else:
        G = A21 / R
    return 'gini', G




#------------------
# xgb
params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}  # using a reasonable values of params


submit=test_id.to_frame()
submit['target']=0

#submit['target_valid']=0

nrounds=2000
kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=0)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100,
                          feval=gini_xgb, maximize=True, verbose_eval=100)
    submit['target'] += xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit+50)/(2*kfold)

 #   submit['target_valid'] += xgb_model.predict(xgb.DMatrix(X_valid),
    # ntree_limit=xgb_model.best_ntree_limit + 50)/(2*kfold)

#    print('AUC = ', roc_auc_score(y_valid, np.ravel(submit['target_valid'])))

# Create a submission file

submit.to_csv('submit_2017_10_13_xgb_data_processed.csv', index = False)










# # PCA on all features - treat cat and bin as num
# from sklearn.decomposition import PCA
#
# # # plotting the pca spectrum
# # pca = PCA()
# # X_all = pd.concat([train_df_linear, test_df_linear])
# # X_all = X_all.drop(['target', 'id'], axis = 1)
# # pca.fit(X_all)
# #
# # plt.figure(1, figsize=(4, 3))
# # plt.clf()
# # plt.axes([.2, .2, .7, .7])
# # plt.plot(pca.explained_variance_, linewidth=2)
# # plt.axis('tight')
# # plt.xlabel('n_components')
# # plt.ylabel('explained_variance_')
# # plt.show()
#
#
# sum_var_threshold= 0.96 # 96% of the variance to be explained
# pca = PCA(n_components=sum_var_threshold, random_state = 42)
# X_all = pd.concat([train_df_linear, test_df_linear])
# X_all = X_all.drop(['target', 'id'], axis = 1)
# pca.fit(X_all)
# print(pca.explained_variance_ratio_)
#
# print(train_df_linear.shape)
#
#
# train_df_linear = train_df_linear.drop(['target', 'id'], axis = 1)
# test_df_linear = test_df_linear.drop(['id'], axis = 1)
#
#
# # uncomment if doing PCA
# # train_df_linear = pca.transform(train_df_linear)
# # test_df_linear = pca.transform(test_df_linear)
#
# #prepare for training and testing
#
# y = train_df['target'].values
# features = pca.components_
# X = train_df_linear
# X_test = test_df_linear
#
# print(train_df_linear.shape)
# print(test_df_linear.shape)
# print(y.shape)
#
#
# # Train and predict using logistic regression
#
# #
# # from sklearn.pipeline import Pipeline
# # from sklearn.model_selection import GridSearchCV
# # from sklearn.linear_model import LogisticRegression
# #
# # from sklearn.metrics import recall_score, make_scorer, roc_auc_score
# # custom_scorer = make_scorer(roc_auc_score)
# #
# # n_components = [0.40, 0.50, 0.60, 0.70] # percentage of variance explained
# # Cs = np.logspace(-6, -2, 4) # C of the logistic regularizer
# #
# # # Parameters of pipelines can be set using ‘__’ separated parameter names:
# #
# # logistic = LogisticRegression(penalty='l1', verbose = 1)
# # pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
# # estimator = GridSearchCV(pipe,
# #                          dict(pca__n_components=n_components,
# #                               logistic__C=Cs), scoring=custom_scorer)
# # estimator.fit(X, y)
# #
# # print('best values of estimators:', estimator.best_estimator_)
#
# from sklearn.linear_model import LogisticRegression
#
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import recall_score, make_scorer, roc_auc_score
#
#
#
#
# from sklearn.model_selection import StratifiedKFold
# kfold = 5
# skf = StratifiedKFold(n_splits=kfold, random_state=0)
# custom_scorer = make_scorer(roc_auc_score)
# logistic = LogisticRegression(penalty='l1', verbose = 1)
# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     print(' kfold: {}  of  {} : '.format(i+1, kfold))
#     X_train, X_valid = X[train_index], X[test_index]
#     y_train, y_valid = y[train_index], y[test_index]
#     fit_train = logistic.fit(X_train, y_train)
#     y_prob_valid = logistic.predict_proba(X_valid)
#     y_pred_valid = int(y_prob_valid >= 0.5)
#     print('AUC = ', roc_auc_score(y_valid, y_pred_valid))
#
#
#  #   submit['target'] += xgb_model.predict(xgb.DMatrix(test[features].values),
#  #                       ntree_limit=xgb_model.best_ntree_limit+50) / (2*kfold)
#
#
#
#
#
#
#
#
# scores = cross_val_score(logistic, X, y, cv=5, scoring=custom_scorer)
# print(scores)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Submitting
#
# # submit=test_df['id'].to_frame()
# # submit['target']= clf.predict_proba(X_test)[:,1]
# # submit.to_csv('submit_2017_10_12_lr.csv', index = False)
#
# #train_df = train_df[num_features].fillna(train_df[num_features].mean())
# #num_features.remove('target')
# #test_df = test_df[num_features].fillna(test_df[num_features].mean())
# #from scipy.stats import mode
# #train_df = train_df[bin_features].fillna(train_df[bin_features].mode())
# #test_df = train_df[bin_features].fillna(train_df[bin_features].mode())
# #train_df = pd.get_dummies(train_df)
# #test_df = pd.get_dummies(test_df)
# #print('Number of NaNs in the entire training data set = ', train_df.isnull().sum().sum())
# #print('Number of NaNs in the entire test data set = ', test_df.isnull().sum().sum())
# #ss = StandardScaler()
# #ss.fit(train_df[num_features])
# #ss.transform(train_df[num_features])
# #ss.transform(test_df[num_features])
#
#
# #train_df = impute_and_scale(train_df, cat_features, bin_features, num_features, print_time = True)
# #test_df = impute_and_scale(test_df, cat_features, bin_features, num_features, print_time = True)
