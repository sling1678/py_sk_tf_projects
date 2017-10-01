import pandas as pd

def read_csvfile_into_df(relative_path_to_csvfile, Index_column):
  df = pd.read_csv(relative_path_to_csvfile)
  df.index = df[Index_column]
  return df
