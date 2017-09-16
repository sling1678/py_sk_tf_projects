#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:25:33 2017

@author: sling
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prelim_explore_data(data):
  print( "\nhead():\n", data.head() )
  print( "\ninfo():\n", data.info() )
  print( "\ndescribe():\n", data.describe() )
  print( "\ndtypes():\n", data.dtypes )
  # for dtype = np.dtype('O') print value_counts()
  for col in data:
    if(data[col].dtype == np.dtype('O')):
      print("\nvalue_counts for", col, ":\n")
      print(data[col].value_counts())
  print("\nLOOK FOR ANY SPECIAL ASPECTS OF DISTRIBUTIONS BELOW\n")
  print("- WHICH FEATURE IS TAIL HEAVY, WHICH HAS DATA CUTOFF ABOVE OR BELOW ETC\n")
  data.hist(bins = 50, figsize = (20, 15))
  plt.show()
  

#-----------------------------
#if __name__ == "__main__":
#  path_to_data_dir = "../datasets/housing"
#  csv_filename = "housing.csv"
#  data = load_data_as_DataFrame_from_csv_file(path_to_data_dir, csv_filename)
#  prelim_explore_data(data)
