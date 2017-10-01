import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def encode_one_hot_one_column(data, cols, replace=False, drop_one=True):
  '''
  A category column is separated into as many columns as there are factors. If drop_one is True
  (default) then one of the creted columns is a default column which is dropped.
  :param data: as DataFrame
  :param cols: as a list of one category column name
  :param replace:
  :param drop_one: if True means return with one less category
  :return: data, default_var, vecData, vec
  '''


  vec = DictVectorizer()
  mkdict = lambda row: dict((col, row[col]) for col in cols)

  vec.fit(data[cols].apply(mkdict, axis=1))
  vecData = pd.DataFrame( vec.transform(data[cols].apply(mkdict, axis=1)).toarray() )
  vecData.columns = vec.get_feature_names()
  vecData.index = data.index
  if replace is True:
    data = data.drop(cols, axis=1)
    data = data.join(vecData)

  if len(vecData.columns) > 1 and drop_one:
    data = data.drop(data.columns[-1], axis=1)
    default_var = vecData.columns[-1]

  return (data, default_var, vecData, vec)

if __name__ == '__main__':
  data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
          'year': [2000, 2001, 2002, 2001, 2002],
          'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}


  df = pd.DataFrame(data)
  df.year = df.year.astype(str)
  for col in [['year'], ['state']]:
    df, default_var, _, _ = encode_one_hot_one_column(df, col, replace=True)
    print(df)
    print('default_var is ',default_var)
