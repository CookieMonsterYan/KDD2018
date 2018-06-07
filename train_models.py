# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:30:49 2018

@author: yaj
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import math
import json

import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.layers import Input, Conv2D, concatenate
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras import backend as K

import models
import utils
import my_loss

def test_score(predicted, actual, plot_file):
    scores=[]
    for i in range(len(predicted)):
        scores.append(utils.smape(actual=actual[i], predicted=predicted[i]))
    plt.figure(figsize=(10,8))
    plt.plot(range(len(scores)),scores,'b-',label='score')
    plt.legend(loc='lower right')
    plt.savefig(plot_file)
    plt.show()
    return scores


def test_model(root, name, model, x, y, norm=None):
    if len(y) > 1:
        predicted,_ = model.predict(x)
        if norm != None:
            predicted = norm(predicted, False)
            y = norm(y[0], False)
    else:
        predicted = model.predict(x)
        if norm != None:
            predicted = norm(predicted, False)
            y = norm(y[0], False)
    
    scores = test_score(predicted, y, os.path.join(root, '{}_test_scores.png'.format(name)))
        
    return predicted, y, scores

def compare_predict_actual(actual, predicted, save_file):
    predicted = predicted.reshape((predicted.shape[0]*predicted.shape[1],))
    y = actual.reshape((actual.shape[0]*actual.shape[1],))
    plt.figure(figsize=(10,8))
    plt.plot(range(len(predicted)),predicted,'b-',label='predicted')
    plt.plot(range(len(y)),y,'r*',label='actual')
    plt.legend(loc='lower right')
    plt.savefig(save_file)
    plt.show()



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def train_model(city, air_code, model_name = 'model_a', train_mother=True, train_child=True, test=False, 
                  special_date=False, special_startday=None,special_endday=None):
    #为模型读取数据
    #####################################################################
    meo_codes=['temperature', 'pressure',	'humidity', 'wind_direction',	'wind_speed/kph']
    aq_stations_file = './data/{}_aq_stations.csv'.format(city.lower())
    aq_data_root = './data/from_aq/'
    meo_data_root = './data/from_grid/'
    #####################################################################
    
    ###############################################################
    window = 1*24
    predict_step = 1*24
    predict_hours = 2*24
    normalized = 2
    file_norm = './data/norm_pars_{}_{}_{}.csv'.format(city.lower(), air_code, model_name)
    folder_norm = os.path.join('./model/',  model_name, city, air_code)
    ##############################################################
    
    if not os.path.isdir(folder_norm):
        os.makedirs(folder_norm)
    
    ########################################################################
    train_rate = 0.8
    ######################################################################
    
    x_stations_obs={}
    x_stations_pre={}
    y_stations={}
    all_data=[]
    total_samples=0
    data_aq_station = pd.read_csv(aq_stations_file, usecols=[1])
    aq_stations = data_aq_station.values
    for station in aq_stations:
        meo_file = os.path.join(meo_data_root, 'from_grid_{}_{}.csv'.format(city.lower(), station[0]))
        aq_file = os.path.join(aq_data_root, 'from_aq_{}_{}.csv'.format(city.lower(), station[0]))
        x_obs=[]
        x_pre=[]
        y=[]
        data_aq = pd.read_csv(aq_file)
        data_aq = data_aq[air_code]
        data_aq = data_aq.values
        data_aq = data_aq.astype('float32')
        data_meo = pd.read_csv(meo_file)
        data_meo = data_meo[meo_codes]
        data_meo = data_meo.values
        data_meo = data_meo.astype('float32')
        len_hours = min(data_aq.shape[0], data_meo.shape[0])
        data_aq = data_aq.reshape((data_aq.shape[0],1))
        data = np.concatenate((data_aq[:len_hours,:], data_meo[:len_hours,:]), axis=1)
        all_data.append(data)
        for i in range(0, (len_hours - window - predict_hours), predict_step):
            if min(data[i:i+window+predict_hours, 0]) < 0:
                continue
            day = i//24
            if special_date:
                if day >= special_startday and day < special_endday:
                    y.append(data[i+window:i+window+predict_hours, 0])
                    x_obs.append(data[i:i+window, :])
                    x_pre.append(data[i+window:i+window+predict_hours, 1:])
                
            else:
                y.append(data[i+window:i+window+predict_hours, 0])
                x_obs.append(data[i:i+window, :])
                x_pre.append(data[i+window:i+window+predict_hours, 1:])
        y = np.array(y)
        x_obs = np.array(x_obs)
        x_pre = np.array(x_pre)
        x_stations_obs[station[0]] = x_obs
        x_stations_pre[station[0]] = x_pre
        y_stations[station[0]] = y
        total_samples += y.shape[0]
        
    data_for_statistic = all_data[0]
    for i in range(1, len(all_data)):
        data_for_statistic = np.concatenate((data_for_statistic,all_data[i]))
    
    means = data_for_statistic.mean(axis=0)
    stds = data_for_statistic.std(axis=0)
    maxs = data_for_statistic.max(axis=0)
    mins = data_for_statistic.min(axis=0)
    with open(file_norm, 'w') as f:
        f.write(',')
        f.write(air_code+',')
        for code in meo_codes:
            f.write(code+',')
        f.write('\n')
        f.write('means:,')
        for mean in means:
            f.write(str(mean)+',')
        f.write('\n')
        f.write('stds:,')
        for std in stds:
            f.write(str(std)+',')
        f.write('\n')
        f.write('maxs:,')
        for max_ in maxs:
            f.write(str(max_)+',')
        f.write('\n')
        f.write('mins:,')
        for min_ in mins:
            f.write(str(min_)+',')
        f.write('\n')
        
    if mins[0] < 0:
        mins[0] = 0
    
    norm_y = utils.normalization(normalized,means[0],stds[0],maxs[0],mins[0])
    norm_x_obs = utils.normalization(normalized,means,stds,maxs,mins)
    norm_x_pre = utils.normalization(normalized,means[1:],stds[1:],maxs[1:],mins[1:])
    
    if train_mother:
        norm_y.save(os.path.join(folder_norm, 'norm_y.json'))
        norm_x_obs.save(os.path.join(folder_norm, 'norm_x_obs.json'))
        norm_x_pre.save(os.path.join(folder_norm, 'norm_x_pre.json'))
    
    
    if special_date:
        for station in aq_stations:
            key = station[0]
            if y_stations[key].shape[0] == 0:
                continue
            days = np.array(range(y_stations[key].shape[0]))
            np.random.shuffle(days)
            x_stations_obs[key] = x_stations_obs[key][days]
            x_stations_pre[key] = x_stations_pre[key][days]
            y_stations[key] = y_stations[key][days]
    
    i=0
    for station in aq_stations:
        key = station[0]
        if y_stations[key].shape[0] == 0:
                continue
        train_row = round(train_rate*y_stations[key].shape[0])
        if i == 0:
            x_train_1 = x_stations_obs[key][:train_row,:,:]
            x_train_2 = x_stations_pre[key][:train_row,:,:]
            y_train = y_stations[key][:train_row,:]
            x_test_1 = x_stations_obs[key][train_row:,:,:]
            x_test_2 = x_stations_pre[key][train_row:,:,:]
            y_test = y_stations[key][train_row:,:]
        else:
            x_train_1 = np.concatenate((x_train_1, x_stations_obs[key][:train_row,:,:]))
            x_train_2 = np.concatenate((x_train_2, x_stations_pre[key][:train_row,:,:]))
            y_train = np.concatenate((y_train, y_stations[key][:train_row,:]))
            x_test_1 = np.concatenate((x_test_1, x_stations_obs[key][train_row:,:,:]))
            x_test_2 = np.concatenate((x_test_2, x_stations_pre[key][train_row:,:,:]))
            y_test = np.concatenate((y_test, y_stations[key][train_row:,:]))
            
        i += 1
    
        
    x_train_1 = norm_x_obs(x_train_1)
    x_train_2 = norm_x_pre(x_train_2)
    y_train = norm_y(y_train)
    x_test_1 = norm_x_obs(x_test_1)
    x_test_2 = norm_x_pre(x_test_2)
    y_test = norm_y(y_test)
    
    
    #模型参数
    ######################################################################
    lr = 1e-5
    batch_size=512
    epoches = 2000
    
    dim_rnn=[256,256,512]
    dim_dense=[512,256,128,64]
    drop=0.2
    activations=['relu','sigmoid']
    
    root_model = os.path.join('./model/', model_name, city, air_code)
    model_structure_file = './model/{}/{}.png'.format(model_name,model_name)
    #####################################################################
    
    if not os.path.isdir(root_model):
        os.mkdir(root_model)
    
    test_total_scores = {}
    
    #-----------------------------------------------------------------------------------------------------
    if train_mother:
        '''
        将所有站点的数据混合在一起
        '''
        input_shape_obs = (window, len(meo_codes)+1)
        input_shape_pre = (predict_hours, len(meo_codes))
        output_shape = predict_hours
        
        optimizer = keras.optimizers.RMSprop(lr=lr)
        loss_fuc = my_loss.loss_smape_rmse
        model = models.model_f(input_shape_obs, input_shape_pre, output_shape, 
                               opt=optimizer, loss=loss_fuc, 
                               dim_rnn=dim_rnn, dim_dense=dim_dense, drop=drop,
                               activations=activations)
        model.summary()
        plot_model(model, to_file=model_structure_file, show_shapes = True, show_layer_names=True)
        
        print('x_train_obs shape:', x_train_1.shape)
        print('x_train_pre shape:', x_train_2.shape)
        print('train samples:', x_train_1.shape[0])
        print('test samples:', x_test_1.shape[0])
        
        hist = model.fit([x_train_1, x_train_2], y_train, 
                  batch_size=batch_size,epochs=epoches,verbose=1,
                  validation_data=([x_test_1,x_test_2], y_test))
        
        model.save(os.path.join(root_model, model_name + '.h5'))
        
        _predicted, _y, scores= test_model(root_model, model_name, model, [x_test_1, x_test_2], [y_test], norm_y)
        
        test_total_scores[model_name] = utils.smape(_y,_predicted)
        print("total_score:{}".format(utils.smape(_y,_predicted)))
        
        compare_predict_actual(_y,_predicted,os.path.join(root_model, '{}_test.png'.format(model_name)))
        
        K.clear_session()
        
    #------------------------------------------------------------------------------------------------------------        
    
    #-----------------------------------------------------------------------------------------------------------------------------------
    '''
    利用使用全部站点数据训练的模型，在每个站点的数据上再分别训练
    '''
    
    if train_child:
        for station in aq_stations:
            s_model_name = model_name+'_'+station[0]
            key = station[0]
            print('*'*5+key+'*'*5)
            
            train_row = round(train_rate*y_stations[key].shape[0])    
            s_x_train_1 = x_stations_obs[key][:train_row,:,:]
            s_x_train_2 = x_stations_pre[key][:train_row,:,:]
            s_y_train = y_stations[key][:train_row,:]
            s_x_test_1 = x_stations_obs[key][train_row:,:,:]
            s_x_test_2 = x_stations_pre[key][train_row:,:,:]
            s_y_test = y_stations[key][train_row:,:]
            
            s_x_train_1 = norm_x_obs(s_x_train_1)
            s_x_train_2 = norm_x_pre(s_x_train_2)
            s_y_train = norm_y(s_y_train)
            s_x_test_1 = norm_x_obs(s_x_test_1)
            s_x_test_2 = norm_x_pre(s_x_test_2)
            s_y_test = norm_y(s_y_test)
            
            
            input_shape_obs = (window, len(meo_codes)+1)
            input_shape_pre = (predict_hours, len(meo_codes))
            output_shape = predict_hours
        
            model = keras.models.load_model( os.path.join(root_model, model_name+'.h5'), custom_objects={'loss_smape_rmse':my_loss.loss_smape_rmse})
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, cooldown=10, min_lr=1e-8)
            early_stopping = EarlyStopping(monitor='val_loss', patience=100)
            model.summary()
            
            print('x_train_obs shape:', s_x_train_1.shape)
            print('x_train_pre shape:', s_x_train_2.shape)
            print('train samples:', s_x_train_1.shape[0])
            print('test samples:', s_x_test_1.shape[0])
            
            hist = model.fit([s_x_train_1, s_x_train_2], s_y_train, 
                      batch_size=batch_size,epochs=epoches,verbose=1,
                      shuffle=True,
                      validation_data=([s_x_test_1,s_x_test_2], s_y_test),
                      callbacks=[reduce_lr,early_stopping])
            
            model.save(os.path.join(root_model, s_model_name + '.h5'))
            
            _predicted, _y, scores= test_model(root_model, s_model_name, model, [s_x_test_1, s_x_test_2], [s_y_test], norm_y)
            
            test_total_scores[s_model_name] = utils.smape(_y,_predicted)
            print("total_score:{}".format(utils.smape(_y,_predicted)))
            
            compare_predict_actual(_y,_predicted,os.path.join(root_model, '{}_test.png'.format(s_model_name)))
            
            K.clear_session()
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    
    #****************************************************************************#
    '''
    保存全部评分
    '''
    with open (os.path.join(root_model,'scores.json'), 'w') as f:
        json.dump(test_total_scores, f)
    
    
    #***************************************************************************#
    
    
    if test:
        test_total_scores = {}
        model = keras.models.load_model( os.path.join(root_model, model_name+'.h5'), custom_objects={'loss_smape_rmse':my_loss.loss_smape_rmse})
        _predicted, _y, scores= test_model(root_model, model_name, model, [x_test_1, x_test_2], [y_test], norm_y)
        
        test_total_scores[model_name] = utils.smape(_y,_predicted)
        print("total_score:{}".format(utils.smape(_y,_predicted)))
        
        K.clear_session()
        
        for station in aq_stations:
            s_model_name = model_name+'_'+station[0]
            key = station[0]
            
            train_row = round(train_rate*y_stations[key].shape[0])    
            s_x_train_1 = x_stations_obs[key][:train_row,:,:]
            s_x_train_2 = x_stations_pre[key][:train_row,:,:]
            s_y_train = y_stations[key][:train_row,:]
            s_x_test_1 = x_stations_obs[key][train_row:,:,:]
            s_x_test_2 = x_stations_pre[key][train_row:,:,:]
            s_y_test = y_stations[key][train_row:,:]
            
            s_x_train_1 = norm_x_obs(s_x_train_1)
            s_x_train_2 = norm_x_pre(s_x_train_2)
            s_y_train = norm_y(s_y_train)
            s_x_test_1 = norm_x_obs(s_x_test_1)
            s_x_test_2 = norm_x_pre(s_x_test_2)
            s_y_test = norm_y(s_y_test)
            
            model = keras.models.load_model( os.path.join(root_model, s_model_name+'.h5'), custom_objects={'loss_smape_rmse':my_loss.loss_smape_rmse})
            _predicted, _y, scores= test_model(root_model, s_model_name, model, [s_x_test_1, s_x_test_2], [s_y_test], norm_y)
            
            test_total_scores[s_model_name] = utils.smape(_y,_predicted)
            print("total_score:{}".format(utils.smape(_y,_predicted)))
            
            K.clear_session()
            
        with open (os.path.join(root_model,'ztest_scores.json'), 'w') as f:
            json.dump(test_total_scores, f)
    
    
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$





if __name__ == "__main__":
    
    citys = ['London']
    air_codes=['PM2.5','PM10']
    for c in citys:
        for a in air_codes:
            print(c)
            print(a)
            test = train_model(c,a,model_name = 'model_f')
    
    
    print('OVER')





