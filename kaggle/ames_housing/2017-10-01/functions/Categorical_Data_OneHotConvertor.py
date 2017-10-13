import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Categorical_Data_OneHotConvertor(BaseEstimator, TransformerMixin):
  def __init__(self, dtype=np.float64, drop_one=True):
    self.dtype = dtype
    self.drop_one = drop_one
  def fit(self, df, y=None):
    """Learn a list of feature name -> indices mappings.
    Parameters
    ----------
    df : DataFrame
    y : (ignored)

    Sets the values of following attributes of the object:

    self.fit_features_

    Returns
    -------
    self
    """
    df_cp = df.copy() # work with a copy
    df_cp = pd.get_dummies(df_cp, drop_first = self.drop_one)
    self.fit_features_ = df_cp.dtypes

    return self

  def transform(self, df, y=None):
    df_cp = df.copy()  # work with a copy
    df_cp = pd.get_dummies(df_cp, drop_first=self.drop_one)
    trf_features = df_cp.dtypes
    trf_col_names = list(trf_features.keys())
    fit_col_names = list(self.fit_features_.keys())
    for feature in fit_col_names:
      if feature not in trf_col_names:
        df_cp[feature] = 0
    df_cp = df_cp[fit_col_names]
    return df_cp




if __name__ == '__main__':
  data = {'col1_str': ['A', 'Z', 'A', 'B', 'B', 'C', 'D'],
          'col2_int': [100, 101, 102, 102, 101, 100, 100],
          'col3_float': [1.1, 1.1, 5.2, 2.3, 2.5, 2.0, 1.1]}

  df = pd.DataFrame(data)
  print(df)

  data2 = {'col1_str': ['A', 'E', 'B'],
          'col2_int': [100, 103, 105 ],
          'col3_float': [1.1, 1.1, 1.0]}

  df2 = pd.DataFrame(data2)
  print(df2)

  # df.col2_int = df.col2_int.astype(str)

  ohc = Categorical_Data_OneHotConvertor()  # set replace=Flse if you want to see the original column
  ohc.fit(df)
  # ff = ohc.fit_features_
  # print(ff)
  # cols = list(ff.keys())
  # print( cols )
  df10 = ohc.transform(df)
  print(df10)
  df20 = ohc.transform(df2)
  print(df20)
