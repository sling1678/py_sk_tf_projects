# IMPUTE MISSING VALUES
# Impute Class
import numpy as np
from sklearn.base import TransformerMixin
class General_Imputer(TransformerMixin):
  def __init__(self, scaling = True):
    """Impute missing values:
    Replace NaN object by 'U'
    Replace NaN in float or int by mean of the column and standard scale if scaling True
    USAGE:
    gim = GeneralImputer()
    gim.fit(X_train)
    gim.transform(X_train)
    gim.transform(X_test)

    """
    self.scaling = scaling

  def fit(self, X, y=None):

    self.replacement_of_objects = 'UNK' # unknown
    self.means = np.mean( X.loc[:, X.dtypes != object])
    self.stds = np.std( X.loc[:, X.dtypes != object])

    return self

  def transform(self, X, y=None):
    X.loc[:, X.dtypes == object] = X.loc[:, X.dtypes == object].fillna(self.replacement_of_objects)
    X.loc[:, X.dtypes != object] = X.loc[:, X.dtypes != object].fillna( self.means )
    if self.scaling:
      X.loc[:, X.dtypes != object] = (X.loc[:, X.dtypes != object] - self.means)/self.stds
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

  print('original df_train:\n', df_train)
  print('original df_test:\n', df_test)
  print('original df_test.dtypes:\n', df_test.dtypes)
  gim = General_Imputer(scaling=True)
  gim.fit(df_train)
  gim.transform(df_train)
  gim.transform(df_test)
  print('modified df_train:\n',  df_train)
  print('modified df_test:\n',  df_test)

