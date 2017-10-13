from .general_imputer import General_Imputer
def impute_all_and_scale_num(X_train, X_test = None):
  gim = General_Imputer(scaling=True)
  gim.fit(X_train)
  X_train = gim.transform(X_train)
  if not X_test.empty:
    X_test = gim.transform(X_test)
  return X_train, X_test

if __name__ == '__main__':
  import pandas as pd
  import numpy as np

  df_train = pd.DataFrame([[1, 1.5, 'A'],
                           [0, 2.5, 'B'],
                           [' ', ' ', ' ']])
  df_test = pd.DataFrame([[1, 1.5, 'X'],
                          [' ', ' ', ' ']])
  df_train = df_train.replace(r'\s+', np.nan, regex=True)
  df_test = df_test.replace(r'\s+', np.nan, regex=True)

  print('original df_train:\n', df_train)
  print('original df_test:\n', df_test)
  print('original df_test.dtypes:\n', df_test.dtypes)

  df_train, df_test = impute_all_and_scale_num(df_train, df_test)
  # gim = General_Imputer(scaling=True)
  # gim.fit(df_train)
  # gim.transform(df_train)
  # gim.transform(df_test)
  print('modified df_train:\n', df_train)
  print('modified df_test:\n', df_test)