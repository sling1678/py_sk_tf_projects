#IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV


# Read csv files
train_csv_file = './datasets/train.csv'
test_csv_file = './datasets/test.csv'
train_data = pd.read_csv(train_csv_file)
train_data.index = train_data.PassengerId
#print(train_data.head(2))
# Randomize training set
permutations = np.random.permutation(train_data.shape[0])
train_data = train_data.iloc[permutations, :]

target_name = 'Survived'
X_train = train_data.drop([target_name, 'PassengerId'],axis = 1)

y_train = pd.DataFrame(train_data[target_name], index = train_data.index)
#print('\nTraining Features:\n', X_train.head(2))
#print('\nTraining Targets:\n',y_train.head(2))
X_test = pd.read_csv(test_csv_file)
X_test.index = X_test.PassengerId
X_test = X_test.drop(['PassengerId'],axis = 1)
#print('\nTest Features:\n',X_test.head(2))

# print(X_train.info())

#------------------------------------------------
# Feature Engineering

# Replace Cabin values by a single letter
X_train['Cabin'] = X_train['Cabin'].str[0].astype(str)
X_test['Cabin'] = X_test['Cabin'].str[0].astype(str)
# Extract titles
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

X_train = extract_titles(X_train, column='Name')
X_test = extract_titles(X_test, column='Name')


#features_to_drop = ['Name', 'Ticket', 'Fare', 'Cabin']
features_to_drop = ['Name', 'Ticket', 'Fare']
# Reasoning:
# Name is unimportant variable other than the title that could be used to aggregate different classes
# the numbering of ticket is already in Pclass
# the effect of Fare is contained in Pclass
# could also drop Cabin since that may also be included in Pclass
X_train = X_train.drop(features_to_drop, axis = 1)
X_test = X_test.drop(features_to_drop, axis = 1)
# print(X_train.info())


# Take care of NaNs - median for floats, most_frequent for ints and objects
# from sklearn.preprocessing import Imputer
# imp_float = Imputer(missing_values='NaN', strategy='median', axis=0) # axis - 0 here means along a column
# imp_int = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# IMPUTE MISSING VALUES
# Impute Class
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
  def __init__(self):
    """Impute missing values.
    Columns of dtype object

    """

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for col in X:
      c1 = pd.DataFrame(X[col], index=X.index)
      if c1.isnull().values.any():
        if c1.dtypes[0] == np.float64:
          c1.fillna(c1.median(), inplace=True)
        else:
          dummy_mode = X[col].value_counts().index[0]
          for idx in c1.index:
            if pd.isnull(c1.loc[idx, col]):
              c1.loc[idx, col] = dummy_mode
              # new_col_name = c1.columns[0]+'_filled'
        new_col_name = c1.columns[0]
        X[new_col_name] = c1

    return X

dfi = DataFrameImputer()
dfi.fit(X_train)
X_train = dfi.transform(X_train)
X_test = dfi.transform(X_test)
# print(X_train.info())
# print(X_train.head(2))


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
columns_to_scale = ['Age']
mms.fit(X_train[columns_to_scale])
X_train['Age_scaled'] = mms.transform(X_train[columns_to_scale].values.reshape(-1, 1))
X_test['Age_scaled'] = mms.transform(X_test[columns_to_scale].values.reshape(-1, 1))
X_train = X_train.drop(columns_to_scale, axis = 1)
X_test = X_test.drop(columns_to_scale, axis = 1)
print(X_train.head(2))

# pd.get_dummies for the objects and integral features
pd_dummies_list = ['Pclass', 'Sex', 'Embarked']
others = ['Age_scaled', 'SibSp', 'Parch']

X_train_cats = pd.get_dummies(X_train[pd_dummies_list])
X_test_cats = pd.get_dummies(X_test[pd_dummies_list])

X_train_others = pd.DataFrame(X_train[others], index = X_train.index)
X_test_others = pd.DataFrame(X_test[others], index = X_test.index)

X_train = pd.concat([X_train_others, X_train_cats], axis=1, join_axes=[X_train_others.index])
X_test = pd.concat([X_test_others, X_test_cats], axis=1, join_axes=[X_test_others.index])

# Thin by dropping one from each categories
extra_cols_to_drop = ['Sex_male', 'Embarked_C']
X_train = X_train.drop(extra_cols_to_drop, axis = 1)
X_test = X_test.drop(extra_cols_to_drop, axis = 1)

set_train = set(X_train.columns)
set_test = set(X_test.columns)
print(set_test - set_train)
assert len(set_test - set_train) == 0

print('\nX_train_cats:\n', X_train_cats.head(2))

print('\nX_train:\n', X_train.head(2))
print('\nX_test:\n', X_test.head(2))

## Models

y_train = np.ravel(y_train)

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

n_neighbors_for_tree = 2
n_estimators_for_RF = 50
models = [LogisticRegressionCV(Cs= 10, penalty = 'l1', solver = 'liblinear'),
          SVC(), neighbors.KNeighborsClassifier(n_neighbors = n_neighbors_for_tree, weights='uniform'),
          tree.DecisionTreeClassifier(), RandomForestClassifier(n_estimators=n_estimators_for_RF),
          GradientBoostingClassifier(n_estimators=n_estimators_for_RF, learning_rate=0.5, max_depth=1, random_state=0)
          ]
for clf in models:
  # print('Model: Regularized Logistic Regression with Cross Validation')
  # clf = LogisticRegressionCV(Cs= 10, penalty = 'l1', solver = 'liblinear')
  clf.fit(X_train, y_train)
  scores = cross_val_score(clf, X_train, y_train, cv=5)
  print('\n', clf, '\n')
  print('scores = ', scores)
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

## Final results
clf = GradientBoostingClassifier(n_estimators=n_estimators_for_RF, learning_rate=0.5, max_depth=1, random_state=0)
clf.fit(X_train, y_train)

scores = cross_val_score(clf, X_train, y_train, cv=5)
print('\n', clf, '\n')
print('scores = ', scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred, index = X_test.index )
y_pred.columns = ['Survived']
print(y_pred.head(5))
submission = pd.DataFrame({
        "PassengerId": y_pred.index,
        "Survived": y_pred['Survived']
    })
submission.to_csv('titanic_grad_boost.csv', index=False)

