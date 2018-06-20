# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:38:46 2018

@author: yaj
"""



import os
import time
import warnings
import numpy as np
from numpy import newaxis
import keras
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Model
from keras.layers import add, multiply
from keras.layers import Input, Conv1D, concatenate, Embedding, BatchNormalization, Flatten, Reshape, TimeDistributed, AveragePooling1D, MaxPooling1D
from keras.utils import plot_model


#LSTM
def model_a(input_rnn1_shape, input_rnn2_shape, output_shape, opt, loss, 
            dim_rnn=[256,256], dim_dense=[512,512,128], drop=0.25,
            activations=['relu','relu','linear']):
    
    inputs_rnn1 = Input(shape=input_rnn1_shape, name='rnn1_input')
    lstm1 = LSTM(dim_rnn[0], activation=activations[0])(inputs_rnn1)
    output_rnn1 = Dense(dim_rnn[0], activation=activations[1],name='rnn1_output')(lstm1)
    
    inputs_rnn2 = Input(shape=input_rnn2_shape, name='rnn2_input')
    lstm2 = LSTM(dim_rnn[1], activation=activations[0])(inputs_rnn2)
    output_rnn2= Dense(dim_rnn[1], activation=activations[1],name='rnn2_output')(lstm2)
    
    x = concatenate([output_rnn1, output_rnn2])
    for dim in dim_dense:
        x = Dense(dim, activation=activations[1])(x)
        x = Dropout(drop)(x)
    main_out = Dense(output_shape, activation=activations[2])(x)
    
    model = Model(inputs=[inputs_rnn1, inputs_rnn2], outputs=main_out)
    model.compile(optimizer=opt, loss=loss)
    
    return model


#encoder-decoder lstm
def model_f(input_rnn1_shape, input_rnn2_shape, output_shape, opt, loss, 
            dim_rnn=[256,256,256], dim_dense=[512,512,128], drop=0.25,
            activations=['relu','linear']):
    
    inputs_rnn1 = Input(shape=input_rnn1_shape, name='rnn1_input')
    lstm1 = LSTM(dim_rnn[0], activation=activations[0])(inputs_rnn1)
    
    inputs_rnn2 = Input(shape=input_rnn2_shape, name='rnn2_input')
    lstm2 = LSTM(dim_rnn[1], activation=activations[0])(inputs_rnn2)
    
    x = concatenate([lstm1, lstm2])
    x = RepeatVector(output_shape)(x)
    
    x = LSTM(dim_rnn[2], return_sequences=True)(x)
    
    for dim in dim_dense:
        x = TimeDistributed(Dense(dim, activation=activations[0]))(x)
        x = TimeDistributed(Dropout(drop))(x)
    main_out = TimeDistributed(Dense(1, activation=activations[1]))(x)
    main_out = Flatten()(main_out)
    
    model = Model(inputs=[inputs_rnn1, inputs_rnn2], outputs=main_out)
    model.compile(optimizer=opt, loss=loss)
    
    return model


