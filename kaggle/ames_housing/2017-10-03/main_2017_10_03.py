import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew # the target will be log-transformed and so will be some features

from time import time
import warnings
warnings.filterwarnings('ignore')

# Methods and metrics
from sklearn.metrics.scorer import make_scorer # to use custom_error_function
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Some models we will explore
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# Read the data from the data file
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
train_df.index = train_df.Id
test_df.index = test_df.Id
train_df.drop('Id', axis = 1, inplace=True)
test_df.drop('Id', axis = 1, inplace=True)
print( train_df.head(2) )
print( test_df.head(2) )
# train_df has X and y and test_df has only X. Actually test_df can be used to learn about features
X_all = pd.concat( (train_df.drop('SalePrice', axis = 1), test_df) )
y_train_raw = train_df.SalePrice
print(X_all.shape)
print(y_train_raw.shape)
print(y_train_raw.head(2))
##### OBJECTIVE  --------------------------
#  Objective of this project is to minimize the following error
def error_func(y1, y2):
  error = np.sqrt(  np.sum( ( np.log1p(y1) - np.log1p(y2) ) ** 2  ) / len(y1) )
  return error
# We can implement this function by using y_train = log1p( y_train_raw )
y_train = np.log1p( y_train_raw )
# Objective will now be to minimize
def error_func2(y1, y2):
  error = np.sqrt(  np.sum( (y1 - y2) ** 2  ) / len(y1) )
  return error
# This is same as mean_squared_error in sklearn.metrics
# We will include this measire of error when we train by usign  scoring = "neg_mean_squared_error"

#### DATA PREPROCESSING --------------------------

num_features = X_all.dtypes[ X_all.dtypes != object ].index
print(num_features)
skew_vals = X_all[num_features].apply(lambda x: skew(x.dropna())) #compute skewness
skew_vals_filtered = skew_vals[ skew_vals > 1] # The threshold is arbitrary but large enough
print(skew_vals_filtered) # index of skew_vals_filtered are the names of those features
# log Transform the highly skewed variables
X_all[skew_vals_filtered.index]  = np.log1p( X_all[skew_vals_filtered.index] )
# Check on skewness
skew_vals = X_all[num_features].apply(lambda x: skew(x.dropna())) #compute skewness
skew_vals_filtered = skew_vals[ skew_vals > 1] # The threshold is arbitrary but large enough
print(skew_vals_filtered) # index of skew_vals_filtered are the names of those features

# replace categorical features by one-hot columns - ignoring the NaNs
X_all = pd.get_dummies(X_all)

# Check for NaNs
print('number of feartures with NaN = ', X_all.isnull().any().sum())
# What to do? A simple solution is to impute by mean
X_all = X_all.fillna(X_all.mean())
print('number of feartures with NaN = ', X_all.isnull().any().sum())

#### TRAINING A MODEL --------------------------
# separate X_test and X_train. Note: y_train is already ready for use
X_train = X_all[:train_df.shape[0]] # the top part of X_all is the training part
X_test = X_all[train_df.shape[0]:]
# A function to run cross-validation with error from custom error
custom_scorer = make_scorer(error_func2, greater_is_better = False)
def run_cv_with_custom_error(model, X_train, y_train, custom_scorer, cv):
  result = -cross_val_score(model, X_train, y_train, scoring=custom_scorer, cv=cv)
  return result
# # MODELS:
# # Ridge - it is aregularized linear regression with a parameter alpha
# # Let find the best value for the regularizer alpha

print("Model : Ridge regression")
# alphas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
# tic = time()
# error_results = [ run_cv_with_custom_error(Ridge(alpha=x),\
#                   X_train, y_train, custom_scorer, cv=5).mean() for x in alphas]
# toc = time()
# print('time took = ', toc-tic, 'sec.')
# print(error_results)
# error_results_with_alphas = pd.Series(error_results, index = alphas)
# error_results_with_alphas.plot()
# plt.title("Error vs alpha for Ridge")
# plt.ylabel("Error")
# plt.xlabel("alpha")
# plt.show()
# # shows minimum is 5 <alpha<20. try more alphas in this region
alphas = range(5,20,1)
tic = time()
error_results = [ run_cv_with_custom_error(Ridge(alpha=x),\
                  X_train, y_train, custom_scorer, cv=5).mean() for x in alphas]
toc = time()
print('time took = ', toc-tic, 'sec.')
print(error_results)
error_results_with_alphas = pd.Series(error_results, index = alphas)
error_results_with_alphas.plot()
plt.title("Error vs alpha for Ridge")
plt.ylabel("Error")
plt.xlabel("alpha")
plt.show()
# Lets get the best value of alpha
min_error = error_results_with_alphas.min()
alpha_best = error_results_with_alphas[error_results_with_alphas == min_error].index[0]
print('best alpha value = ', alpha_best)

## Found this to be the best performer with alpha = 12 and error = 0.12734...
## To do better you will have to process the data differently, taking more information into account




## More plots to visualize the result
## Look at the important features - this is present in the coef_ attribute
clf = Ridge(alpha = alpha_best)
clf.fit(X_train, y_train)
feature_importance = pd.Series(clf.coef_, index = X_train.columns).sort_values()
print('coef :', feature_importance)
# A bar plot of some of the most important features is ften desired
matplotlib.rcParams['figure.figsize'] = (14.0, 7.0)
imp_feats = pd.concat([feature_importance.head(10), feature_importance.tail(10)])
imp_feats.plot(kind = 'barh', title = 'Coefficients in Ridge Regression')
plt.show()

## The plot shows some interesting facts. At this point, we should go back and
## redo some of the feature selection steps. Ask: do any of the high importance features
## make sense? Could we any of these be an artefact of data imputation?

## Lets plot predictions against targets for the training set. This will show any outliers in prediciton

y_train_pred = clf.predict(X_train)
matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)

plt.scatter(y_train,y_train_pred)
plt.plot(y_train,y_train, color = 'r')
plt.xlabel('True y_train')
plt.ylabel('Predicted y_train')
plt.show()

## Although plot looks really decent, we have a few points that are way off. We could hunt those
## data points and delete those records to remove their influence on the model. But we will not do that.




## Try Ealstic Net Regression













# ## Try another model, say Lasso, which is a linear model that estimates sparse coefficients.
# # Lasso has a parameter that regularizes L1-normed coefficients. It also reduces the number
# # of variables upon which the solution is dependent.
# # alphas = [0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
# #alphas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# alphas = np.logspace(-6, -3, 20)
# error_results = [ run_cv_with_custom_error(Lasso(alpha=x),\
#                   X_train, y_train, custom_scorer, cv=5).mean() for x in alphas]
# print(error_results)
# error_results_with_alphas = pd.Series(error_results, index = alphas)
# error_results_with_alphas.plot()
# plt.title("Error vs alpha for Lasso")
# plt.ylabel("Error")
# plt.xlabel("alpha")
# plt.show()
# # Lets get the best value of alpha
# min_error = error_results_with_alphas.min()
# alpha_best = error_results_with_alphas[error_results_with_alphas == min_error].index[0]
# print('best alpha value = ', alpha_best)

## There is also Cross-validated LassoCV which goes through all the alphas you put in
## and performs cross-validation on the training data.



# ## Try another model, say SVR
# #Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
# Cs = [0.01, 0.03, 0.1, 0.3, .5, .7, .9, 1.1, 1.3, 1.5, 1.7, 1.9]
# error_results = [ run_cv_with_custom_error(SVR(C=x),\
#                   X_train, y_train, custom_scorer, cv=5).mean() for x in Cs]
# print(error_results)
# error_results_with_Cs = pd.Series(error_results, index = Cs)
# error_results_with_Cs.plot()
# plt.title("Error vs C for SVR")
# plt.ylabel("Error")
# plt.xlabel("C")
# plt.show()
# # Lets get the best value of alpha
# min_error = error_results_with_Cs.min()
# C_best = error_results_with_Cs[error_results_with_Cs == min_error].index[0]
# print('best C value = ', C_best)

# ## Try another model, say DecisionTreeRegressor
# mss = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# error_results = [ run_cv_with_custom_error(DecisionTreeRegressor(min_samples_split=x, max_features='sqrt'),\
#                   X_train, y_train, custom_scorer, cv=5).mean() for x in mss]
# print(error_results)
# error_results_with_mss = pd.Series(error_results, index = mss)
# error_results_with_mss.plot()
# plt.title("Error vs mss for DecisionTreeRegressor")
# plt.ylabel("Error")
# plt.xlabel("mss")
# plt.show()
# # Lets get the best value of alpha
# min_error = error_results_with_mss.min()
# mss_best = error_results_with_mss[error_results_with_mss == min_error].index[0]
# print('best mss value = ', mss_best)


# # Finally, try Neural Network - MLPRegressor with alpha L2-regulararizer
# from sklearn.neural_network import MLPRegressor
#
# alphas = np.logspace(-7, -4, 10)
# tic = time()
# error_results = [ run_cv_with_custom_error(MLPRegressor(hidden_layer_sizes=(20, 20), alpha=x),\
#                   X_train, y_train, custom_scorer, cv=5).mean() for x in alphas]
# toc = time()
# print('time took = ', toc-tic, 'sec.')
# print(error_results)
# error_results_with_alphas = pd.Series(error_results, index = alphas)
# error_results_with_alphas.plot()
# plt.title("Error vs alpha for MLP")
# plt.ylabel("Error")
# plt.xlabel("alpha")
# plt.show()
# # Lets get the best value of alpha
# min_error = error_results_with_alphas.min()
# alpha_best = error_results_with_alphas[error_results_with_alphas == min_error].index[0]
# print('best alpha value = ', alpha_best)