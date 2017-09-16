#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 23:01:18 2017

@author: sveitser
"""

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with median of column. I modified this from ean to median.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

#----------------------------TESTS-------------------
if __name__ == '__main__':
    data = [
        ['a', 1, 2, 'cat'],
        ['b', 2, 1, 'mouse'],
        ['b', 3, 3, 'mouse'],
        [np.nan, np.nan, np.nan, np.nan]
    ]
    
    data2 = [
        ['a', 10, 20, 'z'],
        ['a', 10, 10, 'x'],
        ['b', 20, 20, 'y'],
        [np.nan, np.nan, np.nan, np.nan]
    ]
    X = pd.DataFrame(data)
    X2= pd.DataFrame(data2)
    gi = DataFrameImputer()
    print('fit on:')
    print(X)
    print('transform :')
    print(X2)
    xt = gi.fit(X)
    xt2 = gi.transform(X2)
    print(xt2)
        