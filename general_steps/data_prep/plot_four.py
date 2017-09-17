#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 09:55:12 2017

@author: sling
"""
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

cols  = X_train.shape[1]
attributes = [0, 3, 5, 7, 9]
X = X_train.iloc[:,attributes]
X["sur"] = y_train
X.head(2)
scatter_matrix(X)
