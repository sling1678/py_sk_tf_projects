import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew # the target will be log-transformed and so will be some features

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from time import time
import warnings
warnings.filterwarnings('ignore')

##--------Special Libraries
from imblearn.over_sampling import SMOTE, ADASYN

##---------Local Function Files -------------------------------------
from load_data import load_data
from parse_feature_names import parse_feature_names
from impute_and_scale import impute_and_scale
from compute_feature_importances_RF import compute_feature_importances_RF
from calculate_confusion_matrix import calculate_confusion_matrix
from filter_for_top_features import filter_for_top_features
#-----------------------------

# Read the data from the data file
X_all, y_train, id_test = load_data("./data/train.csv", "./data/test.csv") 

X_train = X_all[:y_train.shape[0]]
X_test = X_all[y_train.shape[0]:]

#code_test_size = 0.2 # Too large data can impede the testing of the code
#_, X_train, _, y_train= train_test_split(X_train, y_train, test_size=code_test_size, random_state=42)
#X_all = pd.concat([X_train, X_test])

# Also all NaN's are coded by value -1
X_all = X_all.replace(-1, np.NaN)
print( 'pct of nans in each feature:\n', 100*( X_all.isnull().sum()/X_all.shape[0]) )
## shows  ps_car_03_cat     69.094264, ps_car_05_cat     44.818377
## Remove them from data since so much of data is missing
X_all.drop(['ps_car_03_cat', 'ps_car_05_cat'], axis = 1)

#from collections import Counter
#print( 'Data types of features = ', Counter(X_all.dtypes.values) )
#
## Need to change the dtype of all variables whose names end in _cat
cat_features, bin_features, num_features = parse_feature_names(X_all)  # We are leaving binary features as 0/1
X_all = impute_and_scale(X_all, cat_features, bin_features, num_features, print_time = True)

## TODO
## Find important features
### Let us do PCA on the numerical features on X_all
# from sklearn.decomposition import PCA
# pca = PCA()
# pca.fit(X_all[num_features])
# print(pca.explained_variance_ratio_)
# [ 0.0992199   0.06623136  0.05307663  0.04549893  0.03948505  0.03866309
#   0.03860905  0.03859906  0.03857609  0.03852203  0.03847681  0.03845833
#   0.03843421  0.03841262  0.03840049  0.03836429  0.03834001  0.03832603
#   0.03827192  0.03674315  0.03279999  0.02932834  0.0280729   0.01650862
#   0.00816463  0.00641648]
# print(pca.singular_values_)
# [ 1959.25793529  1600.75198083  1432.99242835  1326.76136491  1235.97176775
#   1223.03957318  1222.18456982  1222.02635812  1221.66272489  1220.80643614
#   1220.08969283  1219.79665875  1219.41414979  1219.07153463  1218.87907796
#   1218.30442721  1217.91887057  1217.69670289  1216.83688241  1192.28587692
#   1126.49452358  1065.21172015  1042.16351717   799.18547722   562.03124506
#    498.24219392]
## As you can tell nothing can be discarded

### I was going to study correlations among the numerical features - but I will go away from linear analysis and focus on tree-based analysis

## Use RandomForest
# Need to separate out the training set from test set so we can add synthetic data to the training set
#
#print('RandomForest on Original Training Data:')
#X_train = X_all[:y_train.shape[0]]
#X_test = X_all[y_train.shape[0]:]
#
#X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
##train and predict
#importances, std, y_pred_tmp,  exec_time = compute_feature_importances_RF(X_train_tmp, y_train_tmp, X_test_tmp)  
## test
#confusion_matrix= calculate_confusion_matrix(np.ravel( y_test_tmp), np.ravel(y_pred_tmp)) 
#true_neg, false_pos, false_neg, true_pos = confusion_matrix[0,0], confusion_matrix[0,1], confusion_matrix[1,0], confusion_matrix[1,1]
#print(confusion_matrix) 
#print(classification_report(y_test_tmp, y_pred_tmp))
##select features
#top_features = filter_for_top_features(X_train.columns, importances, threshold_imps = 0.95)  



#importances, std, y_pred,  exec_time = compute_feature_importances_RF(X_train, y_train) # takes about 5 min
#TP, FP, FN, TN = calculate_confusion_matrix(y_train, y_pred) ## TP = 573518, FP = 21694, FN = 0, TN = 0 - wrong!
## We get the predictions that all y's are zeros. We do not get any 1's since 1's are only 3.6% of the instances.
## Try SMOTE to balance the classes

print('RandomForest on SMOTED Training Data:')
tic = time()
X_train = X_all[:y_train.shape[0]]
X_test = X_all[y_train.shape[0]:]
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)
toc = time()
print('SMOTE time = ', toc-tic)
## create temporary train and test sets to examine the goodness of fits

X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X_resampled, y_resampled, test_size=0.5, random_state=42)
#train and predict
importances, std, y_pred_tmp,  exec_time = compute_feature_importances_RF(X_train_tmp, y_train_tmp, X_test_tmp)  
# test
confusion_matrix= calculate_confusion_matrix(np.ravel( y_test_tmp), np.ravel(y_pred_tmp)) 
true_neg, false_pos, false_neg, true_pos = confusion_matrix[0,0], confusion_matrix[0,1], confusion_matrix[1,0], confusion_matrix[1,1]
# TP = 573312, FP = 52579, FN = 197, TN = 520939
print(confusion_matrix) 
print(classification_report(y_test_tmp, y_pred_tmp))
#select features
top_features = filter_for_top_features(X_train.columns, importances, threshold_imps = 0.95) # threshold from 0.9 to 0.95same  68 features selected

##Try ADASYN and compare the top_features sets
#tic = time()
#X_resampled2, y_resampled2 = ADASYN().fit_sample(X_train, y_train)
#toc = time()
#print('ADASYN time = ', toc-tic) ## 94 sec for 10% data
#importances, std, y_pred,  exec_time = compute_feature_importances_RF(X_resampled, y_resampled) # 17 min
#confusion_matrix= calculate_confusion_matrix(np.ravel(y_resampled), np.ravel(y_pred)) 
#true_neg, false_pos, false_neg, true_pos = confusion_matrix[0,0], confusion_matrix[0,1], confusion_matrix[1,0], confusion_matrix[1,1]
## TP = 573312, FP = 52579, FN = 197, TN = 520939
#top_features2 = filter_for_top_features(X_train.columns, importances, threshold_imps = 0.95) # threshold from 0.9 to 0.95same  68 features selected 



### I will suppose now we have the most important features and try to train 
# Transform the data to focus only on the top features
#X_all = X_all[top_features]
#X_train = X_all[:y_train.shape[0]]
#X_test = X_all[y_train.shape[0]:]
#
### At this point we will do a cross-validation to get the most-tuned model.
## But I do not have powerful machine, so i will settle for picked values
#
#tic = time()
#
#X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train) # this step has to be redone
#
#
#from sklearn.ensemble import  RandomForestClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import recall_score, precision_score
#X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X_resampled, y_resampled, test_size=0.33, random_state=42)
#
##n_estimators_list = [25, 50, 75, 100, 125, 150]
#n_estimators_list = [100]
#max_depth=8
#min_samples_leaf=4
#max_features=0.2
#n_jobs=-1
#random_state=0 
#verbose = True
#for n_estimators in n_estimators_list:
#    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=n_jobs, random_state=random_state, verbose = verbose)
#    clf.fit(X_train_tmp, y_train_tmp)
#    y_pred_tmp = clf.predict(X_test_tmp)
#    y_pred_proba_tmp = clf.predict_proba(X_test_tmp)
#    confusion_matrix= calculate_confusion_matrix(np.ravel( y_test_tmp), np.ravel(y_pred_tmp))
#    
#    precision = precision_score(np.ravel( y_test_tmp), np.ravel(y_pred_tmp))
#    recall = recall_score(np.ravel( y_test_tmp), np.ravel(y_pred_tmp))
#    
#    print('Results for : n_estimators = ', n_estimators , '\n')
#    print('precision = ', precision, 'recall = ', recall)
#    print(confusion_matrix)
#    print(y_pred_proba_tmp[:10,1])
#toc = time()
#print('time taken for 6 runs = ', toc-tic)



## Having found  the parameters we  use the clf to predict
tic = time()
X_all = X_all[top_features]
X_train = X_all[:y_train.shape[0]]
X_test = X_all[y_train.shape[0]:]
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train) # this step has to be redone
from sklearn.ensemble import  RandomForestClassifier
n_estimators_list = [100]
max_depth=8
min_samples_leaf=4
max_features=0.2
n_jobs=-1
random_state=0 
verbose = True
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=n_jobs, random_state=random_state, verbose = verbose)
# fit on all available data
print('training')
clf.fit(X_train, y_train)
# predict
print('predicting')
y_pred = clf.predict_proba(X_test)[:,1]
toc = time()
print('time taken for full training and prediction = ', toc-tic)

# Create a submission file
submit = pd.DataFrame()
submit['id'] = id_test
submit['target'] = y_pred
submit.to_csv('submit_2017_10_11_SMOTE_RF.csv', index = False)














#------------Extra Stuff from an earlier attempt
#from sklearn.ensemble import ExtraTreesClassifier
#tic = time()
#forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
#forest.fit(X_all[:y_train.shape[0]], y_train)
#importances = forest.feature_importances_
#std = np.std([tree.feature_importances_ for tree in forest.estimators_],  axis=0)
#indices = np.argsort(importances)[::-1]
#toc = time()
#print('time for one forest of 250 trees using ExtraTreesClassifier', toc-tic, 'sec') # 21.6 minutes wow
#print("Feature ranking:")
#
#for f in range(X_all.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
## Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X_all.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X_all.shape[1]), indices)
#plt.xlim([-1, X_all.shape[1]])
#plt.show()
#
#feats = X_all.columns
#idx, total, threshold_imps = 0, 0, 0.90 
## threshold_imp can be adjusted  in a larger loop , but it takes too long in my computer
#while total < threshold_imps:
#    total = total + importances[idx]
#    idx += 1
#important_feats=feats[indices[:idx]]
#X_all = X_all[important_feats]



# 1a. Use PCA on numerical features to thin num features
# 1b. Look at correlations
# 2. Use random forest on all features - use cross-validation to fix parameters
# 3. Use SVC - cross-validation
# 4. Fid the union and intersection of the top 50% of feartures - look at them



#from sklearn.ensemble import RandomForestClassifier
# TODO
# clf = RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None,
#       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
#       max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,
#       min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
#       random_state=None, verbose=0, warm_start=False, class_weight= “balanced_subsample”)


#--- The following is a code from sklearn website that finds ROC curve
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
#                               GradientBoostingClassifier)
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve
# from sklearn.pipeline import make_pipeline
#
#
# # It is important to train the ensemble of trees on a different subset
# # of the training data than the linear regression model to avoid
# # overfitting, in particular if the total number of leaves is
# # similar to the number of training samples
# X_train, X_test, y_train, y_test = train_test_split(X_all[:y_train.shape[0]],y_train, test_size = 0.5)
# X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
#
#
# n_estimators = 50
# random_state = 42
# # Unsupervised transformation based on totally random trees
# rt = RandomTreesEmbedding(max_depth=5, n_estimators=n_estimators, random_state=random_state)
#
# rt_lm = LogisticRegression()
# pipeline = make_pipeline(rt, rt_lm)
# pipeline.fit(X_train, y_train)
# y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
# fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)
#
# # Supervised transformation based on random forests
# rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimators)
# rf_enc = OneHotEncoder()
# rf_lm = LogisticRegression()
# rf.fit(X_train, y_train)
# rf_enc.fit(rf.apply(X_train))
# rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
#
# y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
# fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
#
# grd = GradientBoostingClassifier(n_estimators=n_estimators)
# grd_enc = OneHotEncoder()
# grd_lm = LogisticRegression()
# grd.fit(X_train, y_train)
# grd_enc.fit(grd.apply(X_train)[:, :, 0])
# grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
#
# y_pred_grd_lm = grd_lm.predict_proba(
#     grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
# fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
#
#
# # The gradient boosted model by itself
# y_pred_grd = grd.predict_proba(X_test)[:, 1]
# fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
#
#
# # The random forest model by itself
# y_pred_rf = rf.predict_proba(X_test)[:, 1]
# fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
# plt.plot(fpr_rf, tpr_rf, label='RF')
# plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
# plt.plot(fpr_grd, tpr_grd, label='GBT')
# plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
# plt.plot(fpr_rf, tpr_rf, label='RF')
# plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
# plt.plot(fpr_grd, tpr_grd, label='GBT')
# plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve (zoomed in at top left)')
# plt.legend(loc='best')
# plt.show()












### --------Some Old attempt
# # Figuring out if any columns need to be log-transformed
# skew_vals = X_all[num_features].apply(lambda x: skew(x.dropna())) #compute skewness
# skew_vals_filtered1 = skew_vals[ skew_vals > 1 ] # The threshold is arbitrary but large enough
# # This results in some binary type data to also show up as highly skewed. Eliminate them:
# skew_vals_filtered = skew_vals_filtered1[ skew_vals_filtered1 < 10]
# print(skew_vals_filtered) # index of skew_vals_filtered are the names of those features
#
# # import seaborn as sns
# # sns.set(color_codes=True)
# # sns.distplot( X_all['ps_reg_02'] )
# #
# # # define poisson function, parameter lamb (can't use reserved word lambda) is the fit parameter
# # from scipy.misc import factorial
# # from scipy.optimize import curve_fit
# # def poisson(k, lamb):
# #     return (lamb**k/factorial(k)) * np.exp(-lamb)
# # def exponential(k, lamb):
# #     return lamb * np.exp(-lamb * k)
# # # fit with curve_fit
# # hist = np.histogram(X_all['ps_reg_02'], bins=40, density = True)
# # s = pd.Series( X_all['ps_reg_02'].value_counts() )
# # s = s.sort_index()
# # bin_edges = s.index.ravel()
# # hist =  s.values.ravel()
# # area = 0
# # for i in range(len(bin_edges)):
# #     if i < len(bin_edges) - 1:
# #         area += ( hist[i]  + hist [i+1] ) * ( bin_edges[i+1] - bin_edges[i] ) /2
# #     else:
# #         area += hist[i] * ( bin_edges[i] - bin_edges[i-1] ) /2
# # hist = hist/area
# # params, cov_matrix = curve_fit(exponential, bin_edges, hist)
# #
# # # %matplotlib
# #
# # width = 0.2
# # hist_ = pd.DataFrame( {'histogram' : [bin_edges, hist] ,})
# # fit_ = pd.DataFrame( { 'fit' : [bin_edges, exponential(bin_edges, params)] ,})
# #
# # plt.bar(bin_edges, hist, width = width )
# # plt.plot(bin_edges, exponential(bin_edges, params), color = 'r', lw = 3)
# # plt.show()
# #
# # sns.distplot( X_all['ps_reg_02'] )
#
#
# # replace categorical features by one-hot columns - ignoring the NaNs
#
# X_all = pd.get_dummies(X_all)
#
# # Check for NaNs
# print('number of feartures with NaN = ', X_all.isnull().any().sum())
# # What to do? A simple solution is to impute by mean
# X_all = X_all.fillna(X_all.mean())
# print('number of features with NaN = ', X_all.isnull().any().sum())
# print("number of y's with NaN = ", y_train.isnull().any().sum())
#
# # Is target imbalanced?
# print( 'PCT of zeros in y_train :',  ( ( y_train == 0 ).sum() )/ y_train.shape ) # 0.96
# print( 'PCT of ones in y_train :',  ( ( y_train == 1 ).sum() )/ y_train.shape )  # 0.04
# print('shape of y_train: ', y_train.shape )
#
# print( np.bincount(y_train) ) # array([573518, 21694])

#
# ## Train so that False Positives are minimized
# from sklearn.metrics import precision_recall_curve, average_precision_score
# from sklearn import svm
# from sklearn.model_selection import train_test_split
# from sklearn.grid_search import GridSearchCV
#
# # Choose parameters
# RS = 42  # random_state
# TS = 0.5  # test_size
#
# X, y = X_all[:y_train.shape[0]], y_train
#
#
# # Split into training and test
# tic = time()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TS, random_state=RS)
# toc = time()
# print('time for train_test split = ', toc-tic, 'sec.')
#
# # Create a simple classifier
#
# tic = time()
# clf = svm.LinearSVC(class_weight="balanced", random_state=RS)
#
# param_grid = {'C' : [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],}
# clf = GridSearchCV(clf, param_grid, scoring="recall")
#
# clf.fit(X_train, y_train)
# toc = time()
# print('time for fitting model = ', toc-tic, 'sec.')
#
# y_pred = clf.decision_function(X_test)
#
# precision, recall, _ = precision_recall_curve(y_test, y_pred)
#
# average_precision = average_precision_score(y_test, y_pred)
#
# plt.step(recall, precision, color='b', alpha=0.2, where='post')
# plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
# plt.show()
#
#
# C_best = clf.best_params_  # is 0.001 , which means we should run even smaller Create






