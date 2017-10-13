#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:04:22 2017

@author: sling
"""

def load_data(train_file, test_file, index_col = 'id', target_col= 'target', print_time = True):
  import pandas as pd
  from time import time
  
  tic = time()
  train_df = pd.read_csv(train_file)
  test_df = pd.read_csv(test_file)

  train_df.index = train_df.id
  test_df.index = test_df.id
  
  id_test = test_df.id
  train_df.drop(index_col, axis = 1, inplace=True)
  test_df.drop(index_col , axis = 1, inplace=True)
  X_all = pd.concat( (train_df.drop(target_col, axis = 1), test_df) )
  y_train = train_df.target
  
  toc = time()
  if print_time:
    print('time to read data into DataFrame = ', toc-tic, 'sec.') ## 16.8 sec
  return X_all, y_train, id_test
