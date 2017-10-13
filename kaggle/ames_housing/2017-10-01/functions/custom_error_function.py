import numpy as np
def custom_error_function(y, y_pred):
  ''' Error function for SalePrice for kaggle Ames housing competition'''
  error = np.sqrt( np.sum(  ( y - y_pred ) ** 2 ) / len(y) )
  return error

if __name__ == '__main__':
  y = np.array([1, 1 ])
  y_pred = np.array([3, 1])

  print( 1.4 < custom_error_function(y, y_pred) < 1.5 ) # should print True