#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:19:57 2017

@author: sling
"""

import numpy as np
import pandas as pd
from sklearn import tree
from data_imputer import DataFrameImputer
from sklearn.preprocessing import LabelBinarizer, Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_copy = X_train.copy()
all_cols = X_copy.columns
imputer = DataFrameImputer()
X_copy = pd.DataFrame(imputer.fit_transform(X_copy))
X_copy.columns = all_cols


all_dtypes = X_copy.dtypes
cat_cols =[]
num_cols = []
for col_name, val in all_dtypes.iteritems():
  if val == 'object':
    cat_cols.append(col_name)
  else:
    num_cols.append(col_name)
X_train_num = X_train[num_cols]

X_train_cat = X_train[cat_cols]
Lb = LabelBinarizer()
X_train_cat_1hot = pd.DataFrame( Lb.fit_transform(X_train_cat), columns = Lb.classes_ )
 



#
#X_copy_cat = X_copy.select_dtypes(include=['object'])
#cat_cols = X_copy_cat.columns
#X_copy_num = X_copy.drop(cat_cols, axis=1)
#
#
#
#from sklearn.base import BaseEstimator, TransformerMixin
#
#class DataFrameSelector(BaseEstimator, TransformerMixin):
#    def __init__(self, attribute_names):
#        self.attribute_names = attribute_names
#    def fit(self, X, y=None):
#        return self
#    def transform(self, X):
#        return X[self.attribute_names].values
#
#num_pipeline = Pipeline([
#        ('imputer', Imputer(strategy="median")),
##        ('attribs_adder', CombinedAttributesAdder()),
#        ('std_scaler', StandardScaler()),
#    ])
#cat_pipeline = Pipeline([
#        ('selector', DataFrameSelector(cat_attribs)),
#        ('label_binarizer', LabelBinarizer()),
#    ])
#
#
#

#columns = X_train.columns
#
#imputer = DataFrameImputer()
#X_copy = pd.DataFrame(imputer.fit_transform(X_copy))
#X_copy.columns = columns
#
#
#
###
#Lb = LabelBinarizer()
#xx = pd.DataFrame(Lb.fit_transform(obj_cols))
###
##clf = tree.DecisionTreeRegressor()
##clf = clf.fit(X_copy, y_train)