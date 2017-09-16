#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:54:16 2017

@author: sling
"""
# Run this file with appropriate links
import os
from six.moves import urllib
import numpy as np
import pandas as pd

from fetch_data_from_web import fetch_and_save_file
from extract_file import extract_tgz
#from load_data import load_data_as_DataFrame_from_csv_file
from prelim_explore_data import prelim_explore_data
#from create_train_valid_test_sets import create_train_valid_test_sets
from generate_working_datasets import load_data_as_DataFrame_from_csv_file
from generate_working_datasets import generate_working_datasets, save_datasets

## CUSTOMIZE THESE

download = False

if download:  
  DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
  src_file_link = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
  path_to_data_dir = os.path.join("..","datasets", "housing")
  filename= "housing.tgz"
  fetch_and_save_file(src_file_link, path_to_data_dir, filename, force = False)

if not download:
  path_to_data_dir = os.path.join("..","datasets", "housing")
  filename= "housing.tgz"

label_list = ["median_house_value"]

## DO NOT CHANGE THESE
tgz_dir = path_to_data_dir
tgz_filename = filename
extract_tgz(tgz_dir, tgz_filename, path_to_data_dir, force = False)

csv_filename = ''.join(filename.split('.')[0:-1]) + ".csv"
data = load_data_as_DataFrame_from_csv_file(path_to_data_dir, csv_filename)

datasets_all, datasets_sample = generate_working_datasets(path_to_data_dir, csv_filename, label_list)

save_datasets(path_to_data_dir, csv_filename, datasets_all, is_sample = False)
save_datasets(path_to_data_dir, csv_filename, datasets_sample, is_sample = True)
##Now you will have data saved as pickle files in the directory chosen 

# Just some preliminary
prelim_explore_data(data)