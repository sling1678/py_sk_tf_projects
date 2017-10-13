#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:28:33 2017

@author: sling
"""
def filter_for_top_features( all_features, importances, threshold_imps = 0.95):
    import numpy as np
    idx, total, threshold_imps = 0, 0, threshold_imps
    indices = np.argsort(importances)[::-1]
    while total < threshold_imps:
        total = total + importances[idx]
        idx += 1
    top_features=all_features[indices[:idx]]
    return top_features