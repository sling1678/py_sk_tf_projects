
import pandas as pd

def print_info_about_data(train_data, test_data = None):

  if not train_data.empty:
    print('\nShape of train_data: \n')
    print( train_data.shape )

    print( train_data.head() )

    print( 'Info on train_data:')
    print( train_data.info() )

    print(' \nDexcription of train_data: \n' )
    print( train_data.describe() )

  if not test_data.empty:

      print( '\nDexcription of test_data: \n')
      print( test_data.describe())

      print('\nShape of test_data: \n')
      print( test_data.shape )

      print('\nInfo on test_data:\n')
      print(test_data.info())
      tr_f = dict( train_data.dtypes[train_data.dtypes == float] )
      tr_i = dict( train_data.dtypes[train_data.dtypes == int] )
      tr_o = dict( train_data.dtypes[train_data.dtypes == object] )
      te_f = dict( test_data.dtypes[test_data.dtypes == float] )
      te_i = dict( test_data.dtypes[test_data.dtypes == int] )
      te_o = dict( test_data.dtypes[test_data.dtypes == object] )
      print('\nDatatypes of train_data: \n')
      print( tr_f.keys() - te_f.keys() )
      print( te_f.keys() - tr_f.keys()  )
      print( tr_i.keys() - te_i.keys() )
      print( te_i.keys() - tr_i.keys() )
