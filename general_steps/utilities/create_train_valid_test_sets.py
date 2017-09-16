#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:08:04 2017

@author: sling
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def create_train_valid_test_sets(data, test_size = 0.2, valid_size = 0.2):
  """
  Split the full data set into three subsets 
  Returns the DataFrames train, valid, test in that order 
  """
  
  train_set_full, test = train_test_split(data, test_size=test_size, random_state=42)
  train, valid = train_test_split(train_set_full, test_size=valid_size, random_state=42)
  
  return train, valid, test

#----------------------------------------------
#if __name__ == "__main__":
#  test_data = [ [1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9], 
#        [10, 11, 12],
#        [13, 14, 15],
#        [16, 17, 18],
#        [19, 20, 21], 
#        [22, 23, 24],
#        [25, 26, 27],
#        [28, 29, 30]]
#  test_data = pd.DataFrame(test_data)
#  test_data.columns = ["A", "B", "C"]
#  col_to_maintain_balance = "B"
#  print(test_data)
#  
#  train, valid, test = create_train_valid_test_sets(test_data)
#  print("\nTrian Set:")
#  print(train)
#  print("\nValid Set:")
#  print(valid)
#  print("\nTest Set:")
#  print(test)
#  