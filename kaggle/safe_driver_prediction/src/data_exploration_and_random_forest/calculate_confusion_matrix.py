#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:36:00 2017

@author: sling
"""


#def calculate_confusion_matrix(y_true, y_pred, positive=0):
#    TP, FP, FN, TN = 0, 0, 0, 0
#    for i in range(len(y_true)):
#        if y_true[i] == positive:
#            if y_pred[i] == positive:
#                TP += 1
#            else:
#                FN += 1
#        else:
#            if y_pred[i] == positive:
#                    FP += 1
#            else:
#                    TN += 1
#    return TP, FP, FN, TN

def calculate_confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred,  labels=None, sample_weight=None)


if __name__ == '__main__':
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1]
    C = calculate_confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = C[0,0], C[0,1], C[1,0], C[1,1]
    print( TP == 4, FN == 1, FP == 2, TN == 6)
    print(C)
    
        