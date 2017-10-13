def print_df(df, get_shape = True, get_head = True, get_info = True,\
             get_describe = True):
  if df is not None:
    #name = [x for x in globals() if globals()[x] is df][0]

    if get_shape:
      print('\nShape of', df.name, ':\n', df.shape)

    if get_info:
      print( '\nInfo  of', df.name, ':\n')
      print( df.info() )

    if get_head:
      print( '\nhead() of', df.name, ':\n', df.head(1) )

    if get_describe:
      print('\nDescription  of', df.name, ':\n', df.describe() )
  else:
    print('No data to print')

def print_info_about_data(train_data, test_data = None, get_shape = True, get_head = True, get_info = True,\
             get_describe = True):
  print_df(train_data, get_shape, get_head, get_info, get_describe)
  print_df(test_data, get_shape, get_head, get_info, get_describe)

if __name__ == '__main__':

  import pandas as pd
  import numpy as np

  df_train = pd.DataFrame( [ [ 1, 1.5, 'A'],
                       [0, 2.5, 'B'],
                       [' ', ' ',' ' ]])
  df_test = pd.DataFrame( [ [ 1, 1.5, 'X'],
                            [ ' ', ' ', ' ']])
  df_train = df_train.replace(r'\s+', np.nan, regex=True)
  df_test = df_test.replace(r'\s+', np.nan, regex=True)


  df_train.name = 'df_train'
  df_test.name = 'df_test'

  print('INFO:')
  print(df_train.info())

  print_df(df_train)
  print_info_about_data(df_train, df_test)
