#  Reference
# https://www.kaggle.com/apapiu/regularized-linear-models

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from functions.load_data import load_data as load_data

# STEP 1: READ IN THE DATA
train_csvfile, test_csvfile = '../data/train.csv', '../data/test.csv'
Index_column, Target_column = 'Id', 'SalePrice'

train, y_train, test = load_data(train_csvfile, test_csvfile, Index_column, Target_column)
# print_info_about_data(X_train, X_test, get_shape = True,\
#                       get_head = False, get_describe = False, get_info = False)

# print(X_train.head(2))
#print(y_train.head())

all_data = pd.concat((train, test))
print(all_data.shape)
# Data preprocessing:

# We're not going to do anything fancy here:
#
# First I'll transform the skewed numeric features by taking log(feature + 1)
# - this will make the features more normal
# Create Dummy variables for the categorical features
# Replace the numeric missing values (NaN's) with the mean of their respective columns

# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# prices = pd.DataFrame({"price":y_train["SalePrice"], "log(price + 1)":np.log1p(y_train["SalePrice"])})
# prices.hist()

#log transform the target:
y_train["SalePrice"] = np.log1p(y_train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = y_train.SalePrice

# Models
#
# Now we are going to use regularized linear regression models
# from the scikit learn module.
# I'm going to try both l_1(Lasso) and l_2(Ridge) regularization. I'll also
# define a function that returns the cross-validation
# rmse error so we can evaluate our models and pick the best tuning par

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean()
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

#Note the U-ish shaped curve above. When alpha is too large the regularization
# is too strong and the model cannot capture all the complexities in the data.
# If however we let the model be too flexible (alpha small) the model begins to
# overfit. A value of alpha = 10 is about right based on the plot above.
print( cv_ridge.min() )  # 0.127

# So for the Ridge regression we get a rmsle of about 0.127
# Let' try out the Lasso model. We will do a slightly different approach here
# and use the built in Lasso CV to figure out the best alpha for us.
# For some reason the alphas in Lasso CV are really the inverse or the alphas in Ridge.

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print( rmse_cv(model_lasso).mean() )  #0.123

# Nice! The lasso performs even better so we'll just use this one to predict
# on the test set. Another neat thing about the Lasso is that it does feature
# selection for you - setting coefficients of features it deems unimportant
# to zero. Let's take a look at the coefficients:

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + \
      " variables and eliminated the other " + \
      str(sum(coef == 0)) + " variables")
# Lasso picked 111 variables and eliminated the other 177 variables

# Good job Lasso. One thing to note here however is that the features
# selected are not necessarily the "correct" ones - especially since
# there are a lot of collinear features in this dataset. One idea to
# try here is run Lasso a few times on boostrapped samples and see how
# stable the feature selection is. We can also take a look directly at
# what the most important coefficients are:

imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()

# The most important positive feature is GrLivArea - the above ground area by
# area square feet. This definitely sense. Then a few other location and quality
# features contributed positively. Some of the negative features make less sense
# and would be worth looking into more - it seems like they might come from
# unbalanced categorical variables. Also note that unlike the feature importance
# you'd get from a random forest these are actual coefficients in your model -
# so you can say precisely why the predicted price is what it is. The only issue
# here is that we log_transformed both the target and the numeric features so the
# actual magnitudes are a bit hard to interpret.

#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
plt.show()

#The residual plot looks pretty good.To wrap it up let's predict on the test set
# and submit on the leaderboard:

#Adding an xgboost model:

#Let's add an xgboost model to our linear model to see if we can improve our score:

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot() # will plot train and test errors
plt.show()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)

# Output: XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
#        learning_rate=0.1, max_delta_step=0, max_depth=2,
#        min_child_weight=1, missing=None, n_estimators=360, nthread=-1,
#        objective='reg:linear', reg_alpha=0, reg_lambda=1,
#        scale_pos_weight=1, seed=0, silent=True, subsample=1)

xgb_preds = np.expm1(model_xgb.predict(X_test))  # back to regular prices
lasso_preds = np.expm1(model_lasso.predict(X_test))

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

# Many times it makes sense to take a weighted average of uncorrelated results
# - this usually improves the score although in this case it doesn't help that much.

preds = 0.7*lasso_preds + 0.3*xgb_preds
submission = pd.DataFrame({"Id":test.index, "SalePrice":preds})
submission.to_csv("ridge_sol.csv", index = False)


