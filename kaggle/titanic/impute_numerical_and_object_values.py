# IMPUTE MISSING VALUES
# Impute Class
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
    return self

  def transform(self, X, y=None):
    X.loc[:, df.dtypes == object] = X.loc[:, df.dtypes == object].fillna('U')
    X.loc[:, df.dtypes != object] = X.loc[:, df.dtypes != object].fillna(X.loc[:, df.dtypes != object].mean())
    return X

if __name__ == '__main__':

  import pandas as pd
  import numpy as np

  df = pd.DataFrame( [ [ 1, 1.5, 'A'],
                       [0, 2.5, 'B'],
                       [' ', ' ',' ' ]])
  df = df.replace(r'\s+', np.nan, regex=True)
  print(df)
  gim = General_Imputer()
  gim.fit(df)
  gim.transform(df)
  print(df)

