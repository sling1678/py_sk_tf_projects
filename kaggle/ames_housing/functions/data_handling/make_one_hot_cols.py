import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def make_one_hot_cols(data, cols, replace=False, drop_one=True, standardize = True):
  '''
  A category column is separated into as many columns as there are factors. If drop_one is True
  (default) then one of the creted columns is a default column which is dropped.
  :param data: as DataFrame
  :param cols: as a list of one category column name
  :param replace:
  :param drop_one: if True means return with one less category
  :return: data, default_var, vecData, vec
  '''

  default_var = []


  for col in cols:

    col_to_expand = data[col]
    col_as_dict = []
    for index in range( len(col_to_expand) ):
      col_as_dict.append(dict( {col : col_to_expand[index] } ))
    v = DictVectorizer(sparse=False)
    v_dat = pd.DataFrame( v.fit_transform(col_as_dict) )
    v_dat.columns = v.get_feature_names()
    v_dat.index = data.index


    if replace:
      data = data.drop(col, axis=1)
    data = data.join(v_dat)

    if len(v_dat.columns) > 1 and drop_one:
      data = data.drop(data.columns[-1], axis=1)
      default_var.append( v_dat.columns[-1] )

  return (data, default_var, v_dat, v)

if __name__ == '__main__':
  data = {'col1_str': ['A', 'B', 'A', 'B', 'B'],
          'col2_int': [100, 101, 102, 102, 101],
          'col3_float': [1.1, 1.1, 5.2, 2.3, 2.5]}


  df = pd.DataFrame(data)
  df.col2_int = df.col2_int.astype(str)
  #for col in [['year'], ['state']]:
  cols = ['col1_str', 'col2_int']
  #df, default_var, _, _ = \
  df, default_var, v_dat, v = make_one_hot_cols(df, cols, replace=True)
  print(df)
  print('default_var is ',default_var)