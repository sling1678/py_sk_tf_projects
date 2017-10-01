from functions.load_data import load_data
from functions.make_human_decision_on_datatype_of_each_column import make_human_decision_on_datatype_of_each_column
train_csvfile = '../data/train.csv'
test_csvfile = '../data/test.csv'
Index_column = 'Id'
Target_column = 'SalePrice'
X_train, y_train, X_test = load_data(train_csvfile, test_csvfile, Index_column, Target_column)
print( X_train.head() )
#print( y_train.head() )
#print( X_test.head() )
X_train = make_human_decision_on_datatype_of_each_column(X_train)
#print( X_train.info() )

# def ():
#
#   ''''
#   input: features_list
#   output: numerical_features, categorical_features
#   '''
#
#   numerical_features = []
#   categorical_features = []
#
#   pass



class Imputer_and_Categorical_Data_Convertor():
  '''
  Answer:
   (0) drop a column that has more than (th_nan = 20%) missing
   (1) what to do with missing values in numerical columns (mt_num = 'mean', 'median')
   (2) what to do with missing values in categorical columns (mt_cat = 'unknown', 'most_freq')
   (3) convert categorical columns into one-hot numerical variables
  '''
  pass
def explore_and_reduce():
  '''

  1. For each numerical feature nf_i, check correlation of y with linear and a function f(nf_i)
  [Try quadratic only] - this will find a non-linear regression possibilities and pick
  whichever is higher corrrelation
  2. For every pair of numerical features nf_i and nf_j, calculate correlations of nf_i to nf_j and
  of nf_i * nf_j to y. If c(nf_i, nf_j) > th_corr (=0.8) then keep either nf_i, nf_j, or nf_i*nf_j
  depending on whichever has the highest corr with y.
  3. Do PCA of numerical features and derived numerical features.
  Select number of components using cross-validation.

  IF WE HAVE A NUMBER OF CATEGORICAL FEATURES, BEST TO USE A TREE METHOD AND USE IMPORTANCE

  '''
  pass
