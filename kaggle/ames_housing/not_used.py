# Feature Engineering - Manually
# Look at correlation matrix and decide the best features

# the following is not useful
import matplotlib.pyplot as plt
import seaborn as sns
def show_corr_mat_heatmap(train_data):
  corr_mat = train_data.corr() # Use the full data since it has both X and y and we want corr of y with X
  f, ax = plt.subplots( figsize = (12, 9) )
  sns.heatmap(corr_mat, vmax = 0.8, square = True)
  plt.show()