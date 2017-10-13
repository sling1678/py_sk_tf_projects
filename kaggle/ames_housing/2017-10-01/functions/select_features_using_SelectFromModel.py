import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
# clf = LassoCV()
#
# # Set a minimum threshold of 0.25
#
def select_features(clf, X, y, max_num_features = 2, threshold = 0.1):
  sfm = SelectFromModel(clf, threshold=threshold)
  sfm.fit(X, y)
  n_features = sfm.transform(X).shape[1]

  #print('n_features = ', n_features )
  # Reset the threshold till the number of features equals two.
  # Note that the attribute can be set directly instead of repeatedly
  # fitting the metatransformer.
  while n_features > max_num_features:
    sfm.threshold *= 1.02 # increase by 2% compounded
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]
  # print( X.columns[sfm.get_support()] )
  # print('n_features = ', n_features )
  return X.columns[sfm.get_support()]

from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor
def prepare_data_for_training(X_train, y_train, X_test,\
        do_log_of_y = True, clf = DecisionTreeRegressor(), max_num_features = 50, threshold =  0.05):

  if do_log_of_y:
    y_train_new = np.log(np.ravel(y_train))
  else:
    y_train_new = y_train.copy()

  features_selected = select_features(clf, X_train, y_train_new, \
                  max_num_features=max_num_features, threshold=threshold)
  X_train_new = X_train[features_selected]
  X_test_new = X_test[features_selected]

  return X_train_new, y_train_new, X_test_new