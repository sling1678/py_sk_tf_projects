#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:36:17 2017

@author: sling - modified at several places from ideas in 
(1) Hands-On Machine Learning with Scikit-Learn and TensorFlow
By: Aurélien Géron, and (2) Udacity/Deep Learning
"""

import sys # will use sys.stdout.write and sys.stdout.flush methods
import os

from urllib.request import Request, urlopen, urlretrieve
from urllib.error import URLError

def download_progress_hook(block_count, block_size, file_size, last_percent_reported = 0, pct_to_report = 5):
  """A hook to report the progress of a download. Reports every 5% change in download progress.
  """
#  global last_percent_reported
  percent = int(block_count * block_size * 100 / file_size)

  if last_percent_reported != percent: # the signal for Done Downloading
    if percent % pct_to_report == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
  return last_percent_reported

def fetch_file_helper(src_file_link, dest_dir, filename, expected_bytes = None, force = False):
  """
  fethces filename from src_file_link = url/filename and places the copy at dest_dir/filename.
  - expected_bytes if known can serve as a check on the fidelity of the download
  - force can be used to force the download even when a copy already exists
  - this can be useful if we want a fresh copy downloaded for some reason    
  """
  # make dir if not present
  if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)
    
  dest_filename = os.path.join(dest_dir, filename) # makes full path 
  
  # download only if the dest_filename is not preent or force = True
  
  if force or not os.path.exists(dest_filename):
    urlretrieve(src_file_link, dest_filename, reporthook = download_progress_hook)
  elif os.path.exists(dest_filename):
    print("\nThe file", dest_filename, "already exists and not forced to rewrite.")

  if expected_bytes:
#    dest_filename = os.path.join(dest_dir, filename) 
    stat_info = os.stat(dest_filename)

    if stat_info.st_size == expected_bytes:
      print('\nFound and verified size of ', dest_filename, ' to be ', expected_bytes)
    else:
      raise Exception ('\nFailed to verify size of ' + dest_filename)
  else:
    print('\nunverified size of ', dest_filename)
  
def fetch_and_save_file(src_file_link, dest_dir, filename, force = False):
  
  """
  fethces filename from src_file_link = url/filename and places the copy at dest_dir/filename.
  - force can be used to force the download even when a copy already exists  
  """
  
  request = Request(src_file_link)
  try:
    response = urlopen(request) # try to open and see if there is any problem
  except URLError as e:
    if hasattr(e, 'reason'):
      print('We failed to reach a server.')
      print('Reason: ', e.reason)
    elif hasattr(e, 'code'):
      print('The server couldn\'t fulfill the request.')
      print('Error code: ', e.code)
  else: ## everything is OK - so fetch the file
    expected_bytes = int(response.info()["Content-Length"])
    print("\nexpected_bytes = ", expected_bytes)
    fetch_file_helper(src_file_link, dest_dir, filename, expected_bytes)


#Tests
#if __name__ == "__main__":
#  print('\n' + '-'*80)
#  print("\nTesting download_progress_hook :\n")
#  blockSize = 10
#  totalSize = 1000
#  last_percent_reported =  0
#  pct_to_report = 10
#  for count in range(1,100):
#    last_percent_reported = download_progress_hook(count, blockSize, totalSize,\
#      last_percent_reported, pct_to_report)
#  
#  print('\n' + '-'*80)
#  print("\nTesting fetch_file :")
#  src_file_link = "http://python.org"
#  dest_dir = './data'
#  filename="ex.txt"
#  fetch_and_save_file(src_file_link, dest_dir, filename, force = False)
