#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:11:14 2017

@author: sling
"""

def parse_feature_names(df):
  cat_columns = []
  bin_columns = []
  num_columns = []
  col_names = df.columns
  for col in col_names:
    split_list = col.split('_')
    if 'cat' in split_list:
      cat_columns.append(col)
    elif 'bin' in split_list:
      bin_columns.append(col)
    else:
      num_columns.append(col)

  return cat_columns, bin_columns, num_columns