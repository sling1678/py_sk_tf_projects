#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:15:27 2017

@author: sling
"""

from load_pickle_data import load_pickle_data

## Load Data from appropriate pickle file

IS_SAMPLE = True
 
PICKLE_FILE_ALL_DATA = "../datasets/housing/housing_samples.pickle"
PICKLE_FILE_SAMPLE_DATA = "../datasets/housing/housing_samples.pickle"


# FOR TITANIC DATA
PICKLE_FILE_ALL_DATA = "../../kaggle/titanic/datasets/train_all.pickle"
PICKLE_FILE_SAMPLE_DATA = "../../kaggle/titanic/datasets/train_samples.pickle"



#--------------------------Do not change below  
is_sample = IS_SAMPLE
if is_sample:  
  path_to_pickle_file = PICKLE_FILE_SAMPLE_DATA
else:
  path_to_pickle_file = PICKLE_FILE_ALL_DATA
  
load_pickle_data(path_to_pickle_file, is_sample)
X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickle_data(path_to_pickle_file, is_sample)

print('Training set', X_train.shape, 'Labels set shape', y_train.shape[0])
print('Validation set', X_valid.shape, 'Labels set shape', y_valid.shape[0])
print('Test set', X_test.shape, 'Labels set shape', y_test.shape[0])
   