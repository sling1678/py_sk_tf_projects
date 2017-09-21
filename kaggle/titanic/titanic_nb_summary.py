import os
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


import matplotlib.pyplot as plt

# START WITH FRESH COPY OF X_TRAIN, X_VALID, X_TEST
IS_SAMPLE = False
PICKLE_FILE_ALL_DATA = "./datasets/train_all.pickle"
PICKLE_FILE_SAMPLE_DATA = "./datasets/train_samples.pickle"


# ------------ load pickle data sets

def load_pickle_data(path_to_pickle_file, is_sample=False):
  ''' A utility function to load pickle data'''
  with open(path_to_pickle_file, 'rb') as f:
    saved = pickle.load(f)
    if is_sample:
      X_train = saved["X_sample_train"]
      y_train = saved["y_sample_train"]
      X_valid = saved["X_sample_valid"]
      y_valid = saved["y_sample_valid"]
      X_test = saved["X_sample_test"]
      y_test = saved["y_sample_test"]

    else:
      X_train = saved["X_train"]
      y_train = saved["y_train"]
      X_valid = saved["X_valid"]
      y_valid = saved["y_valid"]
      X_test = saved["X_test"]
      y_test = saved["y_test"]

    del saved  # will free up memory
    return X_train, y_train, X_valid, y_valid, X_test, y_test


is_sample = IS_SAMPLE
if is_sample:
  path_to_pickle_file = PICKLE_FILE_SAMPLE_DATA
else:
  path_to_pickle_file = PICKLE_FILE_ALL_DATA

load_pickle_data(path_to_pickle_file, is_sample)
X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickle_data(path_to_pickle_file, is_sample)
# NOTE: THIS X_TEST IS NOT THE TEST DATA PROVIDED FOR THE PROJECT
print(X_train.head(2))

# -------------- Check the data types of each columns
print(X_train.dtypes)


# Age         float64
# Cabin        object
# Embarked     object
# Fare        float64
# Name         object
# Parch         int64
# Pclass        int64
# Sex          object
# SibSp         int64
# Ticket       object
# -------------- Covert any data types to appropriate data type - Here everything looks appropriate

# Impute Class

class DataFrameImputer(TransformerMixin):
  def __init__(self):
    """Impute missing values.
    Columns of dtype object or int64 are imputed with the most frequent value
    in column.
    Columns of other types are imputed with median of column. I modified this from ean to median.

    """

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for col in X:
      c1 = pd.DataFrame(X[col], index=X.index)
      if c1.isnull().values.any():
        if c1.dtypes[0] == np.float64:
          c1.fillna(c1.median(), inplace=True)
        elif c1.columns[0] == "Cabin":
          for idx in c1.index:
            if pd.isnull(c1.loc[idx, col]):
              c1.loc[idx, col] = 'U'
        else:
          dummy_mode = X[col].value_counts().index[0]
          for idx in c1.index:
            if pd.isnull(c1.loc[idx, col]):
              c1.loc[idx, col] = dummy_mode
              # new_col_name = c1.columns[0]+'_filled'
        new_col_name = c1.columns[0]
        X[new_col_name] = c1
        # X = X.drop(c1.columns[0], axis = 1)

    return X


# # Call DataFrameImputer.fit on X_train and then  DataFrameImputer.transform on all data sets
# dfi = DataFrameImputer()
# X_train_cp = X_train.copy()
# dfi.fit(X_train_cp)
# X_train_cp = dfi.transform(X_train_cp)
# print(X_train_cp.head(2))
# X_test_cp = X_test.copy()
# X_test_cp = dfi.transform(X_test_cp)
# print(X_test_cp.head(2))

# Add Derived Features by using class Derived_Features_Adder that will call various methods
#  that implement each transformation we seek

def replace_cabin_name_by_single_letter(df, cabin='Cabin'):
  '''
   This function maps each Cabin value with the cabin letter
  :param cabin:
  :return: tranformed df
  '''
  df_cp = df.copy()
  df_cp['Cabin'] = df_cp['Cabin'].str[0].astype(str)
  return df_cp

# X_train_cp =  replace_cabin_name_by_single_letter(X_train_cp)
# print('Replace Cabins by Single letter:')
# print(X_train_cp.head(2))


def extract_titles(df, column='Name'):
  df_cp = df.copy()
  # Keep dictionary of titles
  title_dict = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs', 'Ms': 'Mrs',
    'Master': 'Master',
    'Dr': 'Pro', 'Rev': 'Pro',
    'Col': 'Mil', 'Maj': 'Mil', 'Capt': 'Mil',
    'Lady': 'Noble', 'Sir': 'Noble', 'Mlle': 'Noble', 'Mme': 'Noble',
    'Don': 'Noble', 'the Countess': 'Noble', 'Jonkheer': 'Noble',
  }

  df_cp['Title'] = df_cp[column].map(lambda name: name.split(',')[1].split('.')[0].strip())
  df_cp['Title'] = df_cp['Title'].map(title_dict)
  #    title_counts = df['Title'].value_counts()
  return df_cp
# X_train_cp =  extract_titles(X_train_cp)
# print('Etract Titles:')
# print(X_train_cp.head(2))


def add_age_group(df):
  df_cp = df.copy()
  age_groups = ['Baby', 'Child', 'Young', 'Adult', 'Old', 'Very_Old'] # Age group names
  age_bins = [0, 2, 10, 18, 55, 70, 120]
  df_cp['Age_Group'] = pd.cut(df_cp.Age, age_bins, right=False, labels=age_groups)
  return df_cp


# X_train_cp = add_age_group(X_train_cp)
# print('Add Age Groups:')
# print(X_train_cp.head(2))


def replace_ticket_name_by_single_letter(df,
                                         ticket='Ticket'):
  # mapping each Cabin value with the cabin letter
  df_cp = df.copy()
  #df_cp[ticket] = df_cp[ticket].map(lambda t: t[0])
  df_cp[ticket] = df_cp[ticket].str[0].astype(str)
  return df_cp


# X_train_cp = replace_ticket_name_by_single_letter(X_train_cp)
# print('Ticket names:')
# print(X_train_cp.head(2))


def add_family(df):
  df_cp = df.copy()
  df_cp['Family'] = df_cp['Parch'] + df_cp['SibSp']
  return df_cp


# X_train_cp = add_family(X_train_cp)
# print('Family:')
# print(X_train_cp.head(2))


class Derived_Features_Adder(BaseEstimator, TransformerMixin):
  def __init__(self, add_new_features = True):  ## No *arg or **kargs needed
    '''

    :param add_new_features: hyper parameter that can be used to turn off use of thee features
    '''

    add_new_features = add_new_features




  def fit(self, X, y=None):
    return self  ## nothing to fit from X_train; we want to add the same columns to all df

  def transform(self, X, y=None):
    '''

    :param X:
    :param y:
    :return: transformed X
    '''
    # this is where we add new columns or transform content of any column
    # REPLACE CABIN NAME BY SINGLE LETTER - AGGREGATING
    # EXTRACT AND ADD A NEW TITLE COLUMN - ADDING A NEW COLUMN - TITLE
    # DEFINE AGE_GROUP AND ADD AGE_GROUP COLUMN
    # CONVERT TICKET TO SINGLE CHARACTER CATEGORIES - AGGREGATING
    # DEFINE FAMILY ADD FAMILY COLUMN
    # DROP NAME COLUMN
    X_cp = replace_cabin_name_by_single_letter(X)
    X_cp = extract_titles(X_cp)
    X_cp = add_age_group(X_cp)
    X_cp = replace_ticket_name_by_single_letter(X_cp)
    X_cp = add_family(X_cp)
    X_cp = X_cp.drop('Name', axis=1)
    return X_cp

# dfa = Derived_Features_Adder()
# X_train_cp = dfa.fit_transform(X_train_cp)
# print('Derived_Features_Adder:')
# print(X_train_cp.head(2))



# check any NaNs

def list_pct_of_nans(df):
  ''' Check any NaNs in the DataFrame'''
  for col in df:
    print( col, 100*sum(pd.isnull(df[col]))/df.shape[0], '%' )



# TODO: (1) Scale floats, (2) Binarize objects  - Leave ints alone
# TODO: (3) Train and Predict

def get_features_lists_of_separate_dtypes(df):
  ''' separate out features according to dtypes'''
  float_features = df.dtypes == np.float64
  int_features = df.dtypes == np.int64
  obj_features = df.dtypes == np.object
  df_floats = df.loc[:, float_features]
  df_ints = df.loc[:, int_features]
  df_objs = df.loc[:, obj_features]
  return list(df_floats.columns), list(df_ints.columns), list(df_objs.columns)


# a helper class to delect part of DataFrame
class DataFrameSelector(BaseEstimator, TransformerMixin):
  def __init__(self, attribute_names):
    self.attribute_names = attribute_names

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return X[self.attribute_names].values


# a helper class to scale float data types

class FloatDateScaler(BaseEstimator, TransformerMixin):
  def __init__(selfself, scaling=True):
    self.scaling = scaling
  def fit(self, X, y = None):
    return self
  def transform(self, X, y = None):
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    return X_scaled


# Implement both imputer and feature adder

dfi = DataFrameImputer()
dfa = Derived_Features_Adder()

#the data sets to act on
X_train_cp = X_train.copy()
X_valid_cp = X_valid.copy()
X_test_cp = X_test.copy()


# impute : fit and transform the train
print('X_train before imputing:')
print(X_train_cp.head(2))

dfi.fit(X_train_cp)
X_train_cp = dfi.transform(X_train_cp)
print('X_train after imputing:')
print(X_train_cp.head(2))

# impute: transform the validation set
print('X_valid before imputing:')
print(X_valid_cp.head(2))
X_valid_cp = dfi.transform(X_valid_cp)
print('X_valid after imputing:')
print(X_valid_cp.head(2))

# impute: transform the test
print('X_test before imputing:')
print(X_test_cp.head(2))
X_test_cp = dfi.transform(X_test_cp)
print('X_test after imputing:')
print(X_test_cp.head(2))

# Derived features: fit and transform train

X_train_cp = dfa.fit_transform(X_train_cp)
print('After Derived_Features Added:')
print(X_train_cp.head(2))

# Derived features: transform validation
X_valid_cp = dfa.transform(X_valid_cp)
# Derived features: transform test
X_test_cp = dfa.transform(X_test_cp)

# Must impute again since adding new features sometimes also causes some new NANs

print('X_train before imputing:')
print(X_train_cp.head(2))

dfi = DataFrameImputer() # new DataFrameImputer

dfi.fit(X_train_cp)
X_train_cp = dfi.transform(X_train_cp)
print('X_train after imputing:')
print(X_train_cp.head(2))

print('X_valid before imputing:')
print(X_valid_cp.head(2))
X_valid_cp = dfi.transform(X_valid_cp)
print('X_valid after imputing:')
print(X_valid_cp.head(2))

print('X_test before imputing:')
print(X_test_cp.head(2))
X_test_cp = dfi.transform(X_test_cp)
print('X_test after imputing:')
print(X_test_cp.head(2))

# check for any NaNs in the data left - there should be none
print("NANs in the data:")
print("NaNs in train:")
list_pct_of_nans(X_train_cp) # print statement is inside the function
print("NaNs in validation:")
list_pct_of_nans(X_valid_cp) # print statement is inside the function
print("NaNs in test:")
list_pct_of_nans(X_test_cp) # print statement is inside the function

# get columns with different data type for scaling and binarizing
floats, ints, cats = get_features_lists_of_separate_dtypes(X_train_cp)
print('float features :', floats)
print('int features :', ints)
print('cat features :', cats)

# print("X_train before pipeline: ")
# print(X_train_cp.head(2))
# X_train_prepared = perform_final_processing_of_data(X_train_cp)
# print("X_train after pipeline: ")
# print(X_train_prepared.head(2))


def transform_df(df, scaler):
  '''
  Transform train, valid, and test sets according to fit of scaler by the train set alone.
  helps remember the scaling parameters
  :param df:
  :param scaler:
  :return:
  '''
  floats, ints, cats = get_features_lists_of_separate_dtypes(df)
  df_floats = df[floats]
  df_ints = df[ints]
  df_cats = df[cats]

  df1 = pd.DataFrame(scaler.transform(df_floats), index=df_floats.index, columns=floats)
  df2 = pd.get_dummies(df_cats)
  #print(df1[0:2])
  #print(df2[0:2])
  #print(df2.columns)
  df_result = pd.concat([df1, df2, df_ints], axis = 1, join_axes=[df1.index])
  return df_result

def match_features(df1_1, df2_1):
  '''
  Sometimes the training set has more dummies than test and vice-versa,
  but we need the same number. so we drop the extra ones

  :param df1_1:
  :param df2_1:
  :return: the dataframes df1 and df2 that have the same features
  '''
  df1 = df1_1.copy()
  df2 = df2_1.copy()
  set1 = set(df1.columns)
  set2 = set(df2.columns)
  set_to_drop = set2 - set1
  set_to_add_zeros = set1 - set2
  #df1 = df1.drop(list(set1 - set2), axis=1)
  #df2 = df2.drop(list(set2 - set1), axis=1)
  #df1 = df1.loc[:, set1 & set2]
  df2 = df2.drop(set_to_drop, axis = 1)
  for item in list(set_to_add_zeros):
    df2[item] = 0
  return df2


def make_final_training_datasets(df_tr1, df_va1, scaler, combine_validation_with_training = True):
  df_tr = df_tr1.copy()
  df_va = df_va1.copy()

  if combine_validation_with_training:
    df_tr = pd.concat([df_tr, df_va])
    print('df_tr_shape = ', df_tr.shape)
  floats, ints, cats = get_features_lists_of_separate_dtypes(df_tr)
  df_tr_floats = df_tr[floats]
  scaler.fit(df_tr_floats)

  df_tr = transform_df(df_tr, mm_scaler)
# Take care of validation set if it is to be not part of the training set
  if not combine_validation_with_training:
    df_va = transform_df(df_va, mm_scaler)
# match features
    df_va = match_features(df_tr, df_va)


  return df_tr, df_va, scaler

# check the method above
mm_scaler = MinMaxScaler()
X_train_cp, X_valid_cp, scaler_fitted = make_final_training_datasets(X_train_cp, X_valid_cp, mm_scaler, False)



floats, ints, cats = get_features_lists_of_separate_dtypes(X_test_cp)
X_test_cp =  transform_df(X_test_cp, mm_scaler)
#X_test_cp = pd.DataFrame( scaler_fitted.transform(X_test_cp[floats]), index = X_test_cp.index, columns=floats )
X_test_cp = match_features(X_train_cp, X_test_cp)


print('X_train features: ', X_train_cp.columns)
print('X_valid features: ', X_valid_cp.columns)
print('X_test features: ', X_test_cp.columns)
print('X_train = ', X_train_cp.shape)
print('X_valid = ', X_valid_cp.shape)
print('X_test = ', X_test_cp.shape)
print( X_test_cp.head(2) )



# Fitting Models

# Model 1 : Just predict everyone dead

print('Model: Just guess 0 for every case since high probability of not surviving anyway')
y_pred = pd.DataFrame(0, index = y_test.index, columns = y_test.columns)
target_names = ['0', '1']
print(classification_report(y_test, y_pred))

# Model 2 : Just count the y's in the given data and use it to predict
print('Model: Guess 0 if random number is less than prob of 0')
y_net_train = pd.concat([y_train, y_valid])
s = y_net_train.Survived.value_counts()
p = [s[0]/(s[0] + s[1]), s[1]/(s[0]+s[1])]
len_test_set = y_test.shape[0]
yr = [int(np.random.random() > p[0]) for i in range(len_test_set)]
y_pred = pd.DataFrame(yr, index = y_test.index)
print('survival counts are (0 meaning not survived) :')
print(s)
print(classification_report(y_test, y_pred))

#Model 3: Use Logistic Regression

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# class sklearn.linear_model.LogisticRegression(penalty=’l2’,
# dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
# class_weight=None, random_state=None, solver=’liblinear’, max_iter=100,
# multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1)
#
# Choices to be made:
# penalty = 'l1'
# solver = 'liblinear' since we have binary problem, need L1 penalty, and dataset is small
#           Other choices are : 'newton-cg', lbfgs, 'sag', saga' which handle only L2 penalty
# random_state = 42, for reproducibility from run to run
# C = 1/lambda is regularization hyperparameter, originally set to 1, but I should use
# cross-validation to find the appropriate optimized value
#
#class sklearn.linear_model.LogisticRegressionCV(Cs=10, fit_intercept=True, cv=None,
# dual=False, penalty=’l2’, scoring=None, solver=’lbfgs’, tol=0.0001, max_iter=100,
# class_weight=None, n_jobs=1, verbose=0, refit=True, intercept_scaling=1.0,
# multi_class=’ovr’, random_state=None)
# Cs : list of floats | int
# Each of the values in Cs describes the inverse of regularization strength.
# If Cs is as an int, then a grid of Cs values are chosen in a logarithmic scale
# between 1e-4 and 1e4. Like in support vector machines, smaller values specify stronger regularization.


print('Model: Regularized Logistic Regression with Cross Validation')










