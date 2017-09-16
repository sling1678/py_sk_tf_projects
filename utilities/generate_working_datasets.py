#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:51:48 2017

@author: sling
"""

import os

import numpy as np
import pandas as pd

from six.moves import cPickle as pickle
from sklearn.model_selection import train_test_split


def randomize_data(data):
  
  """
  given a DataFrame with each row as one instance of data, returns a DataFrame that is randomized.
  """
  
  permutations = np.random.permutation(data.shape[0])
  shuffled_data = data.iloc[permutations, :]
  return shuffled_data


def load_data_as_DataFrame_from_csv_file(path_to_data_dir, csv_filename):
  csv_path = os.path.join(path_to_data_dir, csv_filename)
  return pd.read_csv(csv_path)


def create_train_valid_test_sets(data, test_size = 0.2, valid_size = 0.2):
  """
  Split the full data set into three subsets 
  Returns the DataFrames train, valid, test in that order 
  """
  
  train_set_full, test = train_test_split(data, test_size=test_size, random_state=42)
  train, valid = train_test_split(train_set_full, test_size=valid_size, random_state=42)
  
  return train, valid, test

def select_samples_from_full(df, cutoff_size = 1000):
  num_samples = df.shape[0]
  if num_samples > cutoff_size:
    sample_size = cutoff_size
  else:
    sample_size = df.shape[0]
  return sample_size

def separate_X_and_y(df, label_list):
  X = df[df.columns.difference(label_list)]
  y = df[label_list]
  return X, y

def generate_working_datasets(path_to_data_dir, csv_filename, label_list):

  data = load_data_as_DataFrame_from_csv_file(path_to_data_dir, csv_filename)
  randomized_data = randomize_data(data)

  train, valid, test = create_train_valid_test_sets(randomized_data, test_size = 0.2, valid_size = 0.2)
  
  sample_train_size = select_samples_from_full(train)
  sample_valid_size = select_samples_from_full(valid)
  sample_test_size = select_samples_from_full(test)
  
  sample_train = train.iloc[:sample_train_size, :]
  sample_valid = valid.iloc[:sample_valid_size, :]
  sample_test = test.iloc[:sample_test_size, :]
   
  # separate the X and y in each set
  
  X_train, y_train = separate_X_and_y(train, label_list)
  X_valid, y_valid = separate_X_and_y(valid, label_list)
  X_test, y_test = separate_X_and_y(test, label_list)
  X_sample_train, y_sample_train = separate_X_and_y(sample_train, label_list)
  X_sample_valid, y_sample_valid = separate_X_and_y(sample_valid, label_list)
  X_sample_test, y_sample_test = separate_X_and_y(sample_test, label_list)

# package for easier handling

  datasets_all = {
      "X_train" : X_train, "y_train" : y_train,
      "X_valid" : X_valid, "y_valid" : y_valid,
      "X_test" : X_test, "y_test" : y_test,
      }
  datasets_sample = {
      "X_sample_train" : X_sample_train, "y_sample_train" : y_sample_train,
      "X_sample_valid" : X_sample_valid, "y_sample_valid" : y_sample_valid,
      "X_sample_test" : X_sample_test, "y_sample_test" : y_sample_test,
      }
 
  return datasets_all, datasets_sample


def save_datasets(path_to_data_dir, csv_filename, datasets, is_sample = False):
  file_name_list = csv_filename.split('.')
  file_name_list_no_csv = file_name_list[0:-1]
  filename = ''.join(file_name_list_no_csv) # drope .csv
  if is_sample:
    filename_pickle = filename +'_samples'+ '.pickle'
  else:
    filename_pickle = filename +'_all'+ '.pickle'
  pickle_file = os.path.join(path_to_data_dir, filename_pickle)
#  print(pickle_file)
  try:
    with open(pickle_file, "wb") as f:
      print("\nSaving pickle file:", pickle_file)
      pickle.dump(datasets, f, pickle.HIGHEST_PROTOCOL)
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise



#-------------------------------------
#if __name__ == "__main__":
#  path_to_data_dir = "../datasets/housing"
#  csv_filename = "housing.csv"
#  label_list = ["median_house_value"]
#  datasets_all, datasets_sample = generate_working_datasets(path_to_data_dir, csv_filename, label_list)
#  
#  save_datasets(path_to_data_dir, csv_filename, datasets_all, is_sample = False)
#  save_datasets(path_to_data_dir, csv_filename, datasets_sample, is_sample = True)