#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:57:47 2017

@author: sling
"""
from six.moves import cPickle as pickle
import pandas as pd

def load_pickle_data(path_to_pickle_file, is_sample = False):
  with open(path_to_pickle_file, 'rb') as f:
    saved = pickle.load(f)
    if is_sample:
      X_train = saved["X_sample_train"]
      y_train = saved["y_sample_train"]
      X_valid = saved["X_sample_valid"]
      y_valid = saved["y_sample_valid"]
      X_test  = saved["X_sample_test"]
      y_test  = saved["y_sample_test"]
 
    else:
      X_train = saved["X_train"]
      y_train = saved["y_train"]
      X_valid = saved["X_valid"]
      y_valid = saved["y_valid"]
      X_test  = saved["X_test"]
      y_test  = saved["y_test"]
        
    del saved # will free up memory
    return X_train, y_train, X_valid, y_valid, X_test, y_test
    
#--------------------TESTS
if __name__ == "__main__":

  is_sample = True  
  path_to_pickle_file = '../datasets/housing/housing_samples.pickle'
  
  if not is_sample:  
    path_to_pickle_file = '../datasets/housing/housing_all.pickle'


  load_pickle_data(path_to_pickle_file, is_sample)
  X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickle_data(path_to_pickle_file, is_sample)
  
  print('Training set', X_train.shape, 'Labels set shape', y_train.shape[0])
  print('Validation set', X_valid.shape, 'Labels set shape', y_valid.shape[0])
  print('Test set', X_test.shape, 'Labels set shape', y_test.shape[0])
  
  # Better tests will require creating simple data and then test on them



