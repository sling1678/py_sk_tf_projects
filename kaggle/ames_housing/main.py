# We call functions from here to do various tasks for making the prediction
# Tasks:
# 1. Read the data
# 2. Decide on features
# 3. Impute and scale data
# 4. Try multiple models
# 5. Select model, fine-tune, and make predictions

#IMPORTS

#Local

from functions.data_handling.read_csvfile_into_df import read_csvfile_into_df
from functions.data_handling.print_info_about_data import print_info_about_data
from functions.data_handling.replace_objects_by_num_columns import replace_objects_by_num_columns, Dummies_Imputer
from functions.data_handling.general_imputer import General_Imputer
from functions.data_handling.reduce_dimensions_using_PCA import reduce_dimensions_using_PCA

from functions.models.custom_loss_function import custom_loss_function


# Libraries
import numpy as np
import pandas as pd
from time import time

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt




tic = time()
train_csvfile = 'data/train.csv'
test_csvfile = 'data/test.csv'
Index_column = 'Id'
Target_column = 'SalePrice'
train_data = read_csvfile_into_df(train_csvfile, Index_column)
test_data = read_csvfile_into_df(test_csvfile, Index_column)

# Some suggest one should Drop outliers

# train_data = train_data[train_data.SalePrice < 400000]
# print('shape of train: ', train_data.shape)


# items that should be object types but are listed as numerical types
int_to_object_list = [ 'MSSubClass', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
#
train_data[int_to_object_list] = train_data[int_to_object_list].astype(object)
test_data[int_to_object_list] = test_data[int_to_object_list].astype(object)
#print_info_about_data(train_data=train_data, test_data=test_data)

corr=train_data.corr()["SalePrice"]
print('corr.shape = ', corr.shape)
features_to_drop = corr[ np.abs( corr[np.argsort(corr, axis=0)[::-1]] ) <  0.2 ].index
print( corr[ np.abs( corr[np.argsort(corr, axis=0)[::-1]] ) <  0.2 ].index)

train_data = train_data.drop(features_to_drop, axis = 1)
test_data = test_data.drop(features_to_drop, axis = 1)

correlations=train_data.corr()
attrs = correlations.iloc[:-1,:-1] # all except target

threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
    .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])),
        columns=['Attribute Pair', 'Correlation'])

    # sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

print( unique_important_corrs )
lst_unique_important_corrs =  list( unique_important_corrs['Attribute Pair'])
redundant_features =[]
for i in unique_important_corrs.index:
  redundant_features.append(unique_important_corrs.loc[i,'Attribute Pair'][1])

train_data = train_data.drop(redundant_features, axis=1)
test_data = test_data.drop(redundant_features, axis=1)

print('train_data.shape : ', train_data.shape)
print('test_data.shape : ', test_data.shape)




# X_train, y_train, X_test DataFrames
#X_train = train_data.drop( [Index_column, Target_column], axis = 1 )
X_train = train_data.drop( [Target_column], axis = 1 )
y_train = train_data[Target_column]
#X_test = test_data.drop([Index_column], axis = 1)
X_test = test_data
# print('X_train shape: ', X_train.shape)
# print('y_train shape: ', y_train.shape)

features_train = set(X_train.columns)
features_test = set(X_test.columns)
print(features_train - features_test)
print( features_test - features_train)
X_train = X_train[sorted(X_train.columns)]
X_test = X_test[sorted(X_test.columns)]

# print( y_train.head() )
# y_train = np.ravel(y_train) # some classifiers need np.array
# print(y_train[:5])

gim = General_Imputer()
print('number of features in train with NaN before fillna = ', pd.isnull(X_train).any().sum())
print('number of features in test with NaN before fillna = ', pd.isnull(X_test).any().sum())
gim.fit(X_train)
X_train = gim.transform(X_train)
X_test = gim.transform(X_test)

print('number of features in training set with NaN after fillna = ', pd.isnull(X_train).any().sum())
print('number of features in test set with NaN after fillna = ', pd.isnull(X_test).any().sum())
if pd.isnull(X_train).any().sum() != 0:
  print('Features with NaN in train still are:\n', X_train.columns[ pd.isnull(X_train).any() ])
if pd.isnull(X_test).any().sum() != 0:
  print('Features with NaN in test still are:\n', X_test.columns[ pd.isnull(X_test).any() ])

## Convert int and floats usign standard scaler

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
#X.loc[:, X.dtypes != object] = X.loc[:, X.dtypes != object].fillna( self.means )
X_train.loc[:, X_train.dtypes != object] = scaler.fit_transform(X_train.loc[:, X_train.dtypes != object])
X_test.loc[:, X_train.dtypes != object] = scaler.transform(X_test.loc[:, X_train.dtypes != object])

# Convert the objects
print('number of features in train before get_dummies = ', len(X_train.columns))
print('number of features in test before get_dummies = ', len(X_test.columns))
X_train = replace_objects_by_num_columns(X_train)
X_test = replace_objects_by_num_columns(X_test)
print('number of features in train after get_dummies = ', len(X_train.columns))
print('number of features in test after get_dummies = ', len(X_test.columns))
dm = Dummies_Imputer()
dm.fit(X_train)
X_train = dm.transform(X_train)
#print(dm.full_cols)
#print(X_train.head())
X_test = dm.transform(X_test)
#print(X_test.head())
print('number of features after get_dummies = ', len(X_train.columns))
print('number of features after get_dummies = ', len(X_test.columns))

# VC dimension argument of how many features to keep
# The VC dimension argument gives an overestimate of N given epsilon, delta, and d_vc
# For error, epsilon <= 0.1 and confidence 1-delta to be 90%, i.e., delta = 0.1
# N approx 10 * d_vc. In linear models, d_vc = num_features. For sampel size N of 1460, we can have 146 features

# Use PCA to select top 146 features

#class sklearn.decomposition.PCA(n_components=None,
#copy=True, whiten=False, svd_solver=’auto’, tol=0.0, iterated_power=’auto’, random_state=None)

toc = time()

print('time for loading and processing data = ', toc-tic, 'sec')

#X_train, X_test = reduce_dimensions_using_PCA(X_train, X_test, threshold_fraction = 0.90)

# print('X-train.shape: ',X_train.shape, 'X_test.shape: ', X_test.shape)
# clf = LinearRegression()
# clf.fit(X_train, np.log(y_train))
# scores = cross_val_score(clf, X_train, np.log(y_train), cv=5)
# print(scores.mean())
# print(scores.std())

scaler_for_y = MinMaxScaler()
y_train_scaled = scaler_for_y.fit_transform(np.log(y_train))

from sklearn.svm import LinearSVR
clf = LinearSVR()
clf.fit(X_train, y_train_scaled)
scores = cross_val_score(clf, X_train, y_train_scaled, cv=5)
print(scores.mean())
print(scores.std())

# from sklearn.ensemble import GradientBoostingRegressor
# clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=10)
# clf.fit(X_train, y_train_scaled)

predicted = cross_val_predict(clf, X_train, y_train_scaled, cv=5)

predicted_log = scaler_for_y.inverse_transform(predicted)
y_train_log = scaler_for_y.inverse_transform(y_train_scaled)
print('error = ', np.sqrt( np.sum( ( predicted_log - y_train_log )**2 ) / len(predicted) ))
fig, ax = plt.subplots()
ax.scatter(y_train_scaled, predicted, edgecolors=(0, 0, 0))
ax.plot([predicted.min(), predicted.max()], [predicted.min(), predicted.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# pca_threshold = np.linspace(0.4, 0.9, 10)
# scores_mean = []
# scores_std = []
# for t in pca_threshold:
#   toc1 = time()
#   X_train, X_test = reduce_dimensions_using_PCA(X_train, X_test, t)
#   toc2 = time()
#   print('time for PCA =', toc2 - toc1, 'sec')
#   clf = LinearRegression()
#   clf.fit(X_train, np.log(y_train))
#   scores = cross_val_score(clf, X_train, np.log(y_train), cv=5)
#   scores_mean.append(scores.mean())
#   scores_std.append(scores.std())
# print(pca_threshold)
# print(scores_mean)
# print(scores_std)
#
# plt.figure(1, figsize=(4, 3))
# plt.plot(pca_threshold, scores_mean, linewidth=2)
# plt.axis('tight')
# plt.xlabel('pca_explained_threshold')
# plt.ylabel('scores_mean')
# plt.show()