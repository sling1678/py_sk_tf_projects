import pandas as pd
def load_data(train_csvfile, test_csvfile = None, Index_column = None, Target_column = None):
  '''
  input: data filenames
  :return: X_full and y_train dataframes
  '''
  Xy_train = pd.read_csv(train_csvfile)
  X_test = pd.read_csv(test_csvfile)
  if Index_column:
    Xy_train.index = Xy_train[Index_column]
    X_test.index = X_test[Index_column]
    Xy_train = Xy_train.drop(Index_column, axis = 1)
    X_test = X_test.drop(Index_column, axis=1)
  if Target_column:
    y_train = pd.DataFrame(Xy_train[Target_column], index = Xy_train.index)
    X_train = Xy_train.drop(Target_column, axis = 1)
  else:
    X_train = Xy_train

# Add a name attribute to the DataFrame for later reference in printing
  X_train.name = 'X_train'
  X_test.name = 'X_test'
  y_train.name = 'y_test'

  return X_train, y_train, X_test

