#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:14:32 2017

@author: sling
"""

import os
import sys
import tarfile

from fetch_data_from_web import fetch_and_save_file

def extract_tgz(tgz_dir, tgz_filename, dest_dir = None, force = False):
  tgz_path = os.path.join(tgz_dir, tgz_filename)
  if not dest_dir or not os.path.exists(dest_dir):
    dest_dir = tgz_dir
    force = True
  print("\nExtacting file", tgz_path)
  if os.path.exists(dest_dir) or force:  
    with tarfile.open(tgz_path) as tar:
      sys.stdout.flush()
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(tar, path=dest_dir)


# Tests to be perfomed
#if __name__ == "__main__" :
#  DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
#  HOUSING_PATH = os.path.join("..","datasets", "housing")
#  HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
#  from six.moves import urllib
#  src_file_link = HOUSING_URL
#  dest_dir = HOUSING_PATH
#  filename= "housing.tgz"
#  fetch_and_save_file(src_file_link, dest_dir, filename, force = False)
#  tgz_dir = dest_dir
#  tgz_filename = filename
#  extract_tgz(tgz_dir, tgz_filename, dest_dir, force = False)
 
