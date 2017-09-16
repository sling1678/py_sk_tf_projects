#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 22:19:40 2017

@author: sling
"""
#import numpy as np
#import pandas as pd

def study_correlations(train, y_column, importance_level = 0.01):
  
  corr_matrix = train.corr() # a correlation object  - ignores NaN and Category columns
  select_important_corrs = abs(corr_matrix[y_column]) > importance_level
  print("\nCorrelations of", y_column, " with abs(corr) > ", importance_level, " :\n")
  print(corr_matrix[y_column].sort_values(ascending = False)[select_important_corrs])
   
##  Pearson with p-value also. This is not important here
#  df = train.iloc[:, :-1]
#  columns = df.columns
#  imputer = Imputer(strategy = "mean", axis=1)
#  df = pd.DataFrame(imputer.fit_transform(df))
#  df.columns = columns
#  
#  columns = df.columns.tolist()
#  correlations = {}
#
#  for col_a, col_b in itertools.combinations(columns, 2):
#    pr = pearsonr(df.loc[:, col_a], df.loc[:, col_b])
#    if pr[1] >= 0.05:
#      correlations[col_a + ' : ' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])
#
#  result = pd.DataFrame.from_dict(correlations, orient='index')
#  result.columns = ['PCC', 'p-value']
#
#  print(result.sort_index())
   

#-------------TESTS
if __name__ == "__main__":
   
  from load_pickle_data import load_pickle_data

  path_to_pickle_file = '../datasets/housing/housing_samples.pickle'
  is_sample = True

  load_pickle_data(path_to_pickle_file, is_sample)
  train, valid, test = load_pickle_data(path_to_pickle_file, is_sample)
  
  y_column = "median_house_value"
  importance_level = 0.0
  y_column = "total_rooms"

  study_correlations(train, y_column, importance_level)