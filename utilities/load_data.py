#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:57:52 2017

@author: sling
"""
import pandas as pd
import os

def load_data_as_DataFrame_from_csv_file(path_to_data_dir, csv_filename):
  csv_path = os.path.join(path_to_data_dir, csv_filename)
  return pd.read_csv(csv_path)


#-------------
#if __name__ == "__main__":
#  path_to_data_dir = "./datasets/housing"
#  csv_filename = "housing.csv"
#  data = load_data_as_DataFrame_from_csv_file(path_to_data_dir, csv_filename)
#  