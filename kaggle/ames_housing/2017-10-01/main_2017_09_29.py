from functions.load_data import load_data as load_data
from functions.print_info_about_data import print_info_about_data
from functions.set_datatypes import set_datatypes
from functions.impute_all_and_scale_num import impute_all_and_scale_num
from functions.Categorical_Data_OneHotConvertor import Categorical_Data_OneHotConvertor
from functions.select_features_using_SelectFromModel import select_features, prepare_data_for_training
from functions.custom_error_function import custom_error_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
import warnings
warnings.filterwarnings('ignore')

# Methods and metrics
from sklearn.metrics.scorer import make_scorer # to use custom_error_function
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Some models we will explore
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# STEP 1: READ IN THE DATA
train_csvfile, test_csvfile = '../data/train.csv', '../data/test.csv'
Index_column, Target_column = 'Id', 'SalePrice'

X_train, y_train, X_test = load_data(train_csvfile, test_csvfile, Index_column, Target_column)
# print_info_about_data(X_train, X_test, get_shape = True,\
#                       get_head = False, get_describe = False, get_info = False)

# print(X_train.head(2))
#print(y_train.head())

# STEP 2: MAKE DECISIONS ABOUT THE DATATYPE OF EACH DATA
X_train = set_datatypes(X_train) # **** User-input required ****

# STEP 3: IMPUTE ALL AND WHILE WE ARE AT IT CENTER AND SCALE THE NUMERICAL COLUMNS
X_train, X_test = impute_all_and_scale_num(X_train, X_test)
#print( 'number of columns that have NaN is: ', pd.isnull(X_train).any().sum() )
# print_info_about_data(X_train, X_test, get_shape = True,\
#                       get_head = False, get_describe = False, get_info = False)

# STEP 4: CONVERT CATEGORICAL DATA TO NUMERICAL COLUMNS USING ONE-HOT-THROW-AWAY-ONE
tic = time()
ohc = Categorical_Data_OneHotConvertor()  # set replace=Flse if you want to see the original column
ohc.fit(X_train)
X_train, X_test = ohc.transform(X_train), ohc.transform(X_test)
toc = time()
# print('X_train.shape = ', X_train.shape, ' X_test.shape = ',  X_test.shape)
# print(' time to do one-hot = ', toc-tic, 'sec.')

# STEP 5: SELECT IMPORTANT FEATURES BY USING SelectFromModel AND LassoCV or some other clf

tic = time()
X_train_final, y_train_final, X_test_final = \
  prepare_data_for_training(X_train, y_train, X_test, clf = LassoCV())

toc = time()
# print('X_train.shape = ', X_train_final.shape, 'y_train.shape = ', y_train_final.shape,\
#       ' X_test.shape = ',  X_test_final.shape)
# print(' time to do feature_selction = ', toc-tic, 'sec.')

# STEP 6: TRAIN AND TEST USING CROSS-VALIDATION ON X_train_final, y_train_final

# First specify the error function we will use

my_scorer = make_scorer(custom_error_function, greater_is_better=False)

# Multiple regressors to process
clf_names = [
  'LinearRegression',
  # 'SVR',
  'GradientBoostingRegressor',
  'RandomForestRegressor'
]
clf_objects = [
  LinearRegression(),
  # GridSearchCV(SVR(kernel='rbf', gamma=0.1),
  #                  scoring=my_scorer,
  #                  cv=5,
  #                  param_grid={"C": [1e0, 1e1, 1e2, 1e3],
  #                              "gamma": np.logspace(-2, 2, 5)}),
  GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,\
                                max_depth=1,random_state=0, loss='ls'),
  RandomForestRegressor(n_estimators= 300,  max_features = "sqrt",\
                        random_state = 0),
]
colors =[ 'r', 'b', 'g', 'k']



fig = plt.figure()
ax1 = fig.add_subplot(111)  #1:Row, 1: Col, 1:subplot_number


for i, clf in enumerate(clf_objects):
  print(clf_names[i], ':\n')
  tic = time()
  clf.fit(X_train_final, y_train_final)
  y_predict = clf.predict(X_train_final)
  print('error =', custom_error_function(y_train_final, y_predict ))
  toc = time()
  print('time for execution= ', toc - tic, 'sec.')

  #plots
  ax1.scatter(y_train_final, y_predict, s=10, c=colors[i], marker="s", label=clf_names[i])
plt.legend(loc='upper left');
plt.show()

# Lets try regularized linear models

# from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
# from sklearn.model_selection import cross_val_score
#
# def rmse(model):
#     rmse= np.sqrt(-cross_val_score(model, X_train_final, y_train_final, scoring="neg_mean_squared_error", cv = 5))
#     return(rmse)
# clf = Ridge() # has one parameter - alpha to fit by cross-validation
# alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
# cv_ridge = [rmse(Ridge(alpha = alpha)).mean()
#             for alpha in alphas]
# cv_ridge = pd.Series(cv_ridge, index = alphas)
# cv_ridge.plot(title = "Validation - Just Do It")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()
# print(cv_ridge.min())

# final run
# clf = RandomForestRegressor(n_estimators=300, max_features="sqrt", \
#                       random_state=0)
# clf.fit(X_train_final, y_train_final)
# y_pred = clf.predict(X_test_final)
#
# # Create contest submission
# submission = pd.DataFrame({
#         "Id": X_test.index,
#         "SalePrice": np.exp(y_pred)
#     })
#
# submission.to_csv('ames_random_forest.csv', index=False)


# SVR
clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                 scoring=my_scorer,
                 cv=5,
                 param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                             "gamma": np.logspace(-2, 2, 5)})
clf.fit(X_train_final, y_train_final)
y_pred = clf.predict(X_test_final)

# Create contest submission
submission = pd.DataFrame({
        "Id": X_test.index,
        "SalePrice": np.exp(y_pred)
    })

submission.to_csv('ames_svr.csv', index=False)