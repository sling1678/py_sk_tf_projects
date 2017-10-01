import pandas as pd
import numpy as np
def replace_objects_by_num_columns(df):
  df = pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
  return df

from sklearn.base import TransformerMixin
class Dummies_Imputer(TransformerMixin):
  def __init__(self):
    """Impute missing values:
    Replace NaN object by 'Unknown'
    Replace NaN in float or int by median
    USAGE:
    dim = Dummies_Imputer()
    dim.fit(X_train)
    dim.transform(X_train)
    dim.transform(X_test)

    """

  def fit(self, X, y=None):
    X = replace_objects_by_num_columns(X)
    self.full_cols = X.columns
    return self

  def transform(self, X, y=None):
    X_new = replace_objects_by_num_columns(X)
    new_cols = X_new.columns
    #print(new_cols)
    full_cols_set = set(self.full_cols)
    new_cols_set = set(new_cols)
    # print(full_cols_set)
    # print(new_cols_set)
    # print(new_cols_set - full_cols_set)
    for col in (new_cols_set - full_cols_set):
      X_new = X_new.drop(col, axis = 1)
    changed_new_col_set = set(X_new.columns)
    for col in (full_cols_set - changed_new_col_set):
      X_new[col] = 0
    return X_new



if __name__ == '__main__':
  import pandas as pd
  import numpy as np
  df_train = pd.DataFrame([[1, 1.5, 'A', 'X'],
                     [0, 2.5, 'B', ' '],
                     [1, 3.5, ' ', 'Z']])
  df_test = pd.DataFrame([[1, 1.5, 'A', 'X'], [0, 2.5, 'C', 'Y']])

  df_train = df_train.replace(r'\s+', np.nan, regex=True)
  df_test = df_test.replace(r'\s+', np.nan, regex=True)

  df_train.loc[:, df_train.dtypes == object] = df_train.loc[:, df_train.dtypes == object].fillna('U')
  df_test.loc[:, df_test.dtypes == object] = df_test.loc[:, df_test.dtypes == object].fillna('U')
  print(df_train)
  print(df_test)

  dm = Dummies_Imputer()
  dm.fit(df_train)
  df_train = dm.transform(df_train)
  print(dm.full_cols)
  print(df_train.head())
  df_test = dm.transform(df_test)
  print(df_test.head())

