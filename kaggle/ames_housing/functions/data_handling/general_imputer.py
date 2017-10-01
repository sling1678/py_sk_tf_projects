# IMPUTE MISSING VALUES
# Impute Class
import numpy as np
from sklearn.base import TransformerMixin
class General_Imputer(TransformerMixin):
  def __init__(self):
    """Impute missing values:
    Replace NaN object by 'Unknown'
    Replace NaN in float or int by median
    USAGE:
    gim = GeneralImputer()
    gim.fit(X_train)
    gim.transform(X_train)
    gim.transform(X_test)

    """

  def fit(self, X, y=None):
    # self.mean_of_floats = np.mean( X.loc[:, X.dtypes == float])
    # self.mean_of_ints = np.mean( X.loc[:, X.dtypes == int] )
    # self.replacement_of_objects = 'U'  #just create another factor rather than use most common
    # for col in X:
    #   if X.dtypes[col] == object:
    #     self.replacement_of_objects = 'U'
    #   else:
    #     self.mean_of_col = np.mean(X.loc[:, col])
    self.replacement_of_objects = 'U'
    self.means = np.mean( X.loc[:, X.dtypes != object])
    self.stds = np.std( X.loc[:, X.dtypes != object])

    return self

  def transform(self, X, y=None):
    # for col in X:
    #   if X.dtypes[col] not in (float, int, object):
    #     print('The data_type is not float, int or object.')
    #     break
    #   else:
    #     if X.dtypes[col] == object:
    #       X.loc[:, col] = X.loc[:, col].fillna(self.replacement_of_objects)
    #     else:
    #       X.loc[:, col] = X.loc[:, col].fillna(self.mean_of_col)
        # X.loc[:, X.dtypes == object] = X.loc[:, X.dtypes == object].fillna(self.replacement_of_objects)
        # X.loc[:, X.dtypes == float] = X.loc[:, X.dtypes == float].fillna( self.mean_of_floats)
        # X.loc[:, X.dtypes == int] = X.loc[:, X.dtypes == int].fillna(self.mean_of_ints)
    X.loc[:, X.dtypes == object] = X.loc[:, X.dtypes == object].fillna(self.replacement_of_objects)
    X.loc[:, X.dtypes != object] = X.loc[:, X.dtypes != object].fillna( self.means )
   # X.loc[:, X.dtypes != object] = (X.loc[:, X.dtypes != object] - self.means)/self.stds
    return X

if __name__ == '__main__':

  import pandas as pd
  import numpy as np

  df_train = pd.DataFrame( [ [ 1, 1.5, 'A'],
                       [0, 2.5, 'B'],
                       [' ', ' ',' ' ]])
  df_test = pd.DataFrame( [ [ 1, 1.5, 'X'],
                            [ ' ', ' ', ' ']])
  df_train = df_train.replace(r'\s+', np.nan, regex=True)
  df_test = df_test.replace(r'\s+', np.nan, regex=True)

  print(df_train)
  print(df_test)
  print(df_test.dtypes)
  gim = General_Imputer()
  gim.fit(df_train)
  gim.transform(df_train)
  gim.transform(df_test)
  print(df_train)
  print(df_test)

