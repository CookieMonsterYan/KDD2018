# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:29:42 2018

自定义loss function

@author: yaj
"""

from keras import backend as K


def loss_smape(y_true,y_pred):
    '''
    smape
    '''
    a = K.abs(y_true - y_pred)
    b = y_true + y_pred
     
    return 2 * K.mean(a/b)


def loss_smape_rmse(y_true,y_pred):
    '''
    smape+RMSE
    '''
    a = K.abs(y_true - y_pred)
    b = y_true + y_pred
     
    smape = 2 * K.mean(a/b)
    
    rmse = K.mean((y_pred-y_true)**2)
    
    return smape + rmse



