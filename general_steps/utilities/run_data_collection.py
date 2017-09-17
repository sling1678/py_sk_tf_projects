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

downloading = False
extracting = False
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
src_file = "datasets/housing/housing.tgz"

DATA_ROOT_IF_NOT_DOWNLOADING = "../datasets/housing"  # CHANGE THIS AS NEEDED
COMPRESSED_DATA_FILE_IF_NOT_DOWNLOADING = "housing.tgz"
LABEL_LIST = ["median_house_value"]

DATA_ROOT_IF_NOT_DOWNLOADING_AND_NOT_EXTRACTING = "../datasets/housing"
CSV_FILENAME_IF_NOT_EXTRACTING = "housing.csv"

## FOR THE TITANIC FILE
DATA_ROOT_IF_NOT_DOWNLOADING_AND_NOT_EXTRACTING = "../../kaggle/titanic/datasets"
CSV_FILENAME_IF_NOT_EXTRACTING = "train.csv"
LABEL_LIST = ["Survived"]


#------------- do not change below

if downloading:
  src_file_link = DOWNLOAD_ROOT + src_file
  src_file_link_list = src_file_link.split('/')
  path_to_data_dir = os.path.join("..",src_file_link_list[-3], src_file_link_list[-2])
  filename= src_file_link_list[-1]  
  fetch_and_save_file(src_file_link, path_to_data_dir, filename, force = False)
else:
  path_to_data_dir = DATA_ROOT_IF_NOT_DOWNLOADING
  filename= COMPRESSED_DATA_FILE_IF_NOT_DOWNLOADING

if extracting:
  tgz_dir = path_to_data_dir
  tgz_filename = filename
  extract_tgz(tgz_dir, tgz_filename, path_to_data_dir, force = False)
  csv_filename = ''.join(filename.split('.')[0:-1]) + ".csv"
else:
  path_to_data_dir = DATA_ROOT_IF_NOT_DOWNLOADING_AND_NOT_EXTRACTING
  csv_filename = CSV_FILENAME_IF_NOT_EXTRACTING 

data = load_data_as_DataFrame_from_csv_file(path_to_data_dir, csv_filename)

label_list = LABEL_LIST
datasets_all, datasets_sample = generate_working_datasets(path_to_data_dir, csv_filename, label_list)
save_datasets(path_to_data_dir, csv_filename, datasets_all, is_sample = False)
save_datasets(path_to_data_dir, csv_filename, datasets_sample, is_sample = True)
##Now you will have data saved as pickle files in the directory chosen 
# Just some preliminary
prelim_explore_data(data)