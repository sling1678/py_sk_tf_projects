
# Simple few feature analysis - based on suggestion by Learning scikit-learn : Machine Learning in Python
# by Raúl Garreta , and Guillermo Moncecchi
# #IMPORTS

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
# permutations = np.random.permutation(train_data.shape[0])
# train_data = train_data.iloc[permutations, :]

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
features_to_keep = ['Pclass', 'Age', 'Sex']
X_train = X_train[features_to_keep]
X_test = X_test[features_to_keep]

# impute Age and any other NaN values
X_train = X_train.fillna(X_train.median())
x_test = X_test.fillna(X_train.median())

#Encode Sex as integers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def change_Sex(df1):
  df = df1.copy()
  df['Sex_float'] = 0
  for idx in df.index:
    val = df.loc[idx, 'Sex']
    if  val == 'male':
      df.loc[idx, 'Sex_float'] = 1.0
    else:
      df.loc[idx, 'Sex_float'] = 0.0
  df = df.drop('Sex', axis = 1)
  return df

X_train = change_Sex(X_train)
X_test = change_Sex(X_test)


def change_Pclass(df1):
  df = df1.copy()
  df['1st'] = 0.0
  df['2nd'] = 0.0
  df['3rd'] = 0.0
  for idx in df.index:
    val = df.loc[idx, 'Pclass']
    if  val == 1:
      df.loc[idx, '1st'] = 1.0
    elif val == 2:
      df.loc[idx, '2nd'] = 1.0
    elif val == 3:
      df.loc[idx, '3rd'] = 1.0
  df = df.drop('Pclass', axis = 1)
  return df

X_train = change_Pclass(X_train)
X_test = change_Pclass(X_test)


print( X_train.head(2), '\n', y_train.head(2))
print(X_train.shape, y_train.shape)

print( X_test.head(2))

# Train a Decision Tree Classifier


from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from sklearn import tree


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







# y_pred = clf.predict(X_test)
# y_pred = pd.DataFrame(y_pred, index = X_test.index )
# y_pred.columns = ['Survived']
# print(y_pred.head(5))
# submission = pd.DataFrame({
#         "PassengerId": y_pred.index,
#         "Survived": y_pred['Survived']
#     })
# submission.to_csv('titanic_grad_boost.csv', index=False)

