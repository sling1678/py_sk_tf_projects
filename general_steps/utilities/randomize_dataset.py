#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:43:23 2017

@author: sling
"""

import numpy as np
import pandas as pd

def randomize_data(data):
  
  """
  given a DataFrame with each row as one instance of data, returns a DataFrame that is randomized.
  """
  
  permutations = np.random.permutation(data.shape[0])
  shuffled_data = data.iloc[permutations, :]
  return shuffled_data


#----------------------------------
#if __name__ == "__main__":
#  test_data = [ [1, 2, 3],
#          [4, 5, 6],
#          [7, 8, 9], 
#          [10, 11, 12],
#          [13, 14, 15]]
#  test_data = pd.DataFrame(test_data)
#  test_data.columns = ["a", "b", "c"]
#  print("\noriginal data:")
#  print(test_data)
#  print("\nrandomized data:")
#  rd = randomize_data(test_data)
#  print( rd )
 