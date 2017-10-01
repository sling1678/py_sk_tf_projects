import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six


class One_Hot_Convertor(BaseEstimator, TransformerMixin):
  def __init__(self, dtype=np.float64, separator="=", replace = True, sparse=True,
               sort=True, drop_one=True):
    self.dtype = dtype
    self.separator = separator
    self.sparse = sparse
    self.sort = sort
    self.drop_one = drop_one
    self.replace = replace

  def fit(self, df, y=None):
    """Learn a list of feature name -> indices mappings.
    Parameters
    ----------
    df : DataFrame
    y : (ignored)
    Returns
    -------
    self
    """
    feature_names = []
    vocab = {}
    default_features = []
    all_features_mult_dict = {col : 1 for col in df.columns}


    for col in df:
      feature_added = False
      features_created_in_this_col = []
      mult = 0

      for index in df.index:
        val = df.loc[index, col]

        if isinstance(val, six.string_types):
          feature = "%s%s%s" % (col, self.separator, val)
          if feature not in vocab:
            vocab[feature] = len(vocab)
            feature_names.append(feature)
            features_created_in_this_col.append(feature)
            mult += 1
            feature_added = True

      if feature_added and len(feature_names) > 1 and self.drop_one:
        default = feature_names.pop()
        features_created_in_this_col.pop()
        del(vocab[default])
        default_features.append(default)

      if feature_added:
    #    print('col = ', col)
        del (all_features_mult_dict[col])
        for f in features_created_in_this_col:
          all_features_mult_dict[f] = mult

    self.feature_names_ = feature_names
    self.vocabulary_ = vocab
    self.default_features_ = default_features

    self.all_features_mult_dict_ = all_features_mult_dict

    return self



  def transform(self, df, y=None):

    # get parameters from the Object.fit() operation
    feature_names = self.feature_names_
    vocab = self.vocabulary_
    default_features = self.default_features_
    all_features_mult_dict = self.all_features_mult_dict_

   # print(feature_names)
    df_cp = df.copy() # make a copy
    for feature in feature_names:
      df_cp[feature] = 0


    for col in df:
      col_expanded = False
      for index in df.index:
        val = df.loc[index, col]
        if isinstance(val, six.string_types):
          feature = "%s%s%s" % (col, self.separator, val)
          if feature in vocab:
            df_cp.loc[index, feature] = 1
            col_expanded = True
      if col_expanded and self.replace:
        df_cp = df_cp.drop(col, axis = 1)

    return df_cp




if __name__ == '__main__':
  data = {'col1_str': ['A', 'B', 'A', 'B', 'B', 'C', 'D'],
          'col2_int': [100, 101, 102, 102, 101, 100, 100],
          'col3_float': [1.1, 1.1, 5.2, 2.3, 2.5, 2.0, 1.1]}


  df = pd.DataFrame(data)
  print(df)
  #df.col2_int = df.col2_int.astype(str)

  ohc = One_Hot_Convertor() # set replace=Flse if you want to see the original column
  ohc.fit(df)
  # print('fm', ohc.feature_names_)
  # print('voc', ohc.vocabulary_)
  # print('def', ohc.default_features_)
  # print('mult', ohc.all_features_mult_dict_)

  df2 = ohc.transform(df)
  print(df2)
  # print(df2.mean())