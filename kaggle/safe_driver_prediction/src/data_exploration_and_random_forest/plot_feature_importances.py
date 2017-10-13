#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:19:46 2017

@author: sling
"""
def plot_feature_importances(importances, std, features)
    # Plot the feature importances of the forest
    
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_all.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_all.shape[1]), indices)
    plt.xlim([-1, X_all.shape[1]])
    plt.show()