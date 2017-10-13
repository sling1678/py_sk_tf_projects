def load_pickle_data(path_to_pickle_file):
  from six.moves import cPickle as pickle
  with open(path_to_pickle_file, 'rb') as f:
    saved = pickle.load(f)
    if is_sample:
      X_train = saved["X_sample_train"]
      y_train = saved["y_sample_train"]
      X_valid = saved["X_sample_valid"]
      y_valid = saved["y_sample_valid"]
      X_test = saved["X_sample_test"]
      y_test = saved["y_sample_test"]

    else:
      X_train = saved["X_train"]
      y_train = saved["y_train"]
      X_valid = saved["X_valid"]
      y_valid = saved["y_valid"]
      X_test = saved["X_test"]
      y_test = saved["y_test"]

    del saved  # will free up memory
    return X_train, y_train, X_valid, y_valid, X_test, y_test