def save_pickled_data(df, pickle_file):
  '''
  Save data in pickle - will be easier to read than csv file each time
  '''
  from six.moves import cPickle as pickle
  try:
    with open(pickle_file, "wb") as f:
      print("\nSaving pickle file:", pickle_file)
      pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise