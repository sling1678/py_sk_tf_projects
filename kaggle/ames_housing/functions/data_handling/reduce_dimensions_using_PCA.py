from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def reduce_dimensions_using_PCA(X_train, X_test, threshold_fraction = 0.90):
  pca = PCA()
  pca.fit(X_train)
  # Plot the PCA spectrum

  sum_exp_var_ratio  = []
  sum = 0
  found = False
  n_comp_ideal_idx = len(pca.explained_variance_ratio_)  # initiliazed to all components
  for i in range(len(pca.explained_variance_ratio_)):
    sum += pca.explained_variance_ratio_[i]
    if sum > threshold_fraction and  not found: # two sigma effect
      n_comp_ideal_idx = i
      found = True
    sum_exp_var_ratio.append(sum)
  print('n_comp_ideal_idx = ', n_comp_ideal_idx)

  # plt.figure(1, figsize=(4, 3))
  # plt.axes([.2, .2, .7, .7])
  # plt.plot(sum_exp_var_ratio, linewidth=2)
  # plt.axis('tight')
  # plt.xlabel('n_components')
  # plt.ylabel('explained_variance_ratio(%)')
  # plt.show()

  if n_comp_ideal_idx == 0:
    n_comp_ideal_idx = 1 # at least one feature

  pca = PCA(n_components=n_comp_ideal_idx, whiten = True) # n_comp_ideal_idx = 144 - takes too long

  X_train = pca.fit_transform(X_train)

  X_test = pca.transform(X_test)



  # from sklearn.linear_model import LogisticRegression
  # from sklearn.pipeline import Pipeline
  # from sklearn.model_selection import GridSearchCV
  #
  # logistic = LogisticRegression()
  # pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
  # # Prediction
  # n_components = [20, 40, 60]
  # Cs = np.logspace(-4, 4, 3)
  #
  # # Parameters of pipelines can be set using ‘__’ separated parameter names:
  # estimator = GridSearchCV(pipe,
  #                          dict(pca__n_components=n_components,
  #                               logistic__C=Cs))
  # estimator.fit(X_train, y_train)
  #
  # plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
  #             linestyle=':', label='n_components chosen')
  # plt.legend(prop=dict(size=12))
  # plt.show()
  return X_train, X_test