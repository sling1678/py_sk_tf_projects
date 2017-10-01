import numpy as np
def custom_loss_function(y, y_pred):
  ''' Error function for SalePrice for kaggle Ames housing competition'''
  error = np.sqrt( np.sum(  ( np.log(y) - np.log(y_pred) )**2 ) / len(y) )
  return error

if __name__ == '__main__':
  y = [1, 1, 2, 1]
  y_pred = [2, 1, 2, 2]

  print( 0.49012 < custom_loss_function(y, y_pred) <0.49013 ) # should print True
