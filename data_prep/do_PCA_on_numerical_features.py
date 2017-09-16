#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:54:03 2017

@author: sling
"""

import numpy as np


from sklearn.preprocessing import Imputer # need to impute for PCA

from sklearn.decomposition import PCA



def do_PCA_on_numerical_features(df):
  df_num_cols = []
  for col in df:
    if(df[col].dtype != np.dtype('O')):
      df_num_cols.append(col)
  df_num = df[df_num_cols]
  df_num_cols = df_num.columns
  
  imputer = Imputer(strategy = 'mean')
  df_num = imputer.fit_transform(df_num)
  
  pca = PCA()
  pca.fit(df_num)
  filter_pca = pca.explained_variance_ratio_>0.1
  top_features = df_num_cols[filter_pca]
  
  
  
  
  
  
  return pca.explained_variance_ratio_, top_features

#--------------TESTS

if __name__ == "__main__":
    
  from load_pickle_data import load_pickle_data

  path_to_pickle_file = '../datasets/housing/housing_samples.pickle'
  is_sample = True

  load_pickle_data(path_to_pickle_file, is_sample)
  train, valid, test = load_pickle_data(path_to_pickle_file, is_sample)
  
#  train_num = do_PCA_on_numerical_features(train)
#  print(train_num.head()) # outputs the correct numerical columns
  pca_explained_variance_ratio_, top_features = do_PCA_on_numerical_features(train)
  print(pca_explained_variance_ratio_)
  print(top_features)