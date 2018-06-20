# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 22:29:47 2018

@author: Cookie-2
"""

import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import keras
from keras import backend as K
import time
import pytz
import datetime

import utils
import my_loss
from utils import request_data,prepare_data
from real_time_data_api import request_real_data, request_real_data_aq


#---------------------------------------------------------------------------------------------------------------------------------------
class RunPredict(object):
    def __init__(self, request_paras, station_infos, prefix_predicted, postfix_predicted, folder, use_caiyun, auto_submit=True):
        self.request_paras = request_paras
        self.station_infos = station_infos
        self.auto_submit = auto_submit
        self.prefix_predicted = prefix_predicted
        self.postfix_predicted = postfix_predicted
        self.folder = folder
        self.use_caiyun = use_caiyun
        
    def predict(self):
        '''
        Everyday prediction
        '''
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
            
        data_predict = {}
        data_predict['test_id'] = []
        data_predict['PM2.5'] = []
        data_predict['PM10'] = []
        data_predict['O3'] = []
        
        tz = pytz.timezone('utc')
        now = datetime.datetime.now(tz)
        now = datetime.datetime(now.year,now.month,now.day,now.hour)
        
        data_folder = request_data(self.request_paras, now, use_caiyun=self.use_caiyun)
        
        for station_id in self.station_infos:
            print('*'*5+station_id+'*'*5)
            station_info = self.station_infos[station_id]
            for i in range(48):
                data_predict['test_id'].append('{}#{}'.format(station_id,i))
            for p in station_info['pollutions']:
                print('-'*5+p+'-'*5)
                x = prepare_data(data_folder, station_id, station_info['model_id'][p], station_info['norm'][p],p,use_caiyun=self.use_caiyun)
                #return x
                #print(datetime.datetime.now())
                model = keras.models.load_model(station_info['model_file'][p], custom_objects={'loss_smape_rmse':my_loss.loss_smape_rmse})
                #print(datetime.datetime.now())
                predicted = model.predict(x)
                #return predicted
                norm_y = utils.normalization()
                norm_y.load(station_info['norm'][p]['norm_y'])
                predicted = norm_y(predicted, forward=False)
                for y in predicted[0]:
                    data_predict[p].append(y)
                    
                K.clear_session()
                
        length=max(len(data_predict['O3']),len(data_predict['PM2.5']),len(data_predict['PM10']))
        if len(data_predict['O3']) < length:
            for i in range(length - len(data_predict['O3'])):
                data_predict['O3'].append(0)
                                    
        predict_file = os.path.join(self.folder, self.prefix_predicted + now.strftime("%Y-%m-%d-%H") + self.postfix_predicted +'.csv')
        pd_predict = pd.DataFrame.from_dict(data_predict)
        pd_predict.to_csv(predict_file, columns=['test_id','PM2.5','PM10','O3'], index=False)
        
        if self.auto_submit:
            utils.submit(predict_file,'submit_'+now.strftime("%Y-%m-%d-%H"),'auto_submit')
            
        return predict_file
            
    def predict_data(self,data_folder,file_name=None):
        '''
        predic using sepcial data
        '''
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
            
        data_predict = {}
        data_predict['test_id'] = []
        data_predict['PM2.5'] = []
        data_predict['PM10'] = []
        data_predict['O3'] = []
        
        tz = pytz.timezone('utc')
        now = datetime.datetime.now(tz)
        now = datetime.datetime(now.year,now.month,now.day,now.hour)
        
        #data_folder = request_data(self.request_paras, now)
        
        for station_id in self.station_infos:
            print('*'*5+station_id+'*'*5)
            station_info = self.station_infos[station_id]
            for i in range(48):
                data_predict['test_id'].append('{}#{}'.format(station_id,i))
            for p in station_info['pollutions']:
                print('-'*5+p+'-'*5)
                x = prepare_data(data_folder, station_id, station_info['model_id'][p], station_info['norm'][p],p,use_caiyun=self.use_caiyun)
                #return x
                #print(datetime.datetime.now())
                model = keras.models.load_model(station_info['model_file'][p], custom_objects={'loss_smape_rmse':my_loss.loss_smape_rmse})
                #print(datetime.datetime.now())
                predicted = model.predict(x)
                #return predicted
                norm_y = utils.normalization()
                norm_y.load(station_info['norm'][p]['norm_y'])
                predicted = norm_y(predicted, forward=False)
                for y in predicted[0]:
                    data_predict[p].append(y)
                #del model
                K.clear_session()
                
        length=max(len(data_predict['O3']),len(data_predict['PM2.5']),len(data_predict['PM10']))
        if len(data_predict['O3']) < length:
            for i in range(length - len(data_predict['O3'])):
                data_predict['O3'].append(0)
                
        if file_name == None:
            predict_file = os.path.join(self.folder, self.prefix_predicted+ now.strftime("%Y-%m-%d-%H") + self.postfix_predicted +'.csv')
        else:
            predict_file = os.path.join(self.folder, self.prefix_predicted+ file_name + self.postfix_predicted +'.csv')
        pd_predict = pd.DataFrame.from_dict(data_predict)
        pd_predict.to_csv(predict_file, columns=['test_id','PM2.5','PM10','O3'], index=False)
        
        if self.auto_submit:
            utils.submit(predict_file,'submit_'+now.strftime("%Y-%m-%d-%H"),'auto_submit')
            
        return predict_file
    
    
    def test_standard(self, predict_file, predict_time):
        '''
        Calc scores based on the official evaluation
        '''
        
        root = os.path.join(os.path.split(predict_file)[0],'test')
        predict_file_name = os.path.splitext(os.path.split(predict_file)[1])[0]
        forecast_hours = self.request_paras['forecast']
        
        #live data
        bien = self.request_paras['bien']
        request_real_data_aq(bien['cities'], forecast_hours, root, predict_time)
        utils.clip_real_data_aq(root, self.request_paras['stations'], now=predict_time, backward_hours=forecast_hours, interpolate=False)
        p_aq_codes = {'PM2.5':'PM25_Concentration','PM10':'PM10_Concentration','O3':'O3_Concentration'}
        obs={}
        for station_id in self.station_infos:
            station_info = self.station_infos[station_id]
            aq_file = os.path.join(root,'bien','aq',station_id+'.csv')
            pd_aq = pd.read_csv(aq_file)
            obs_station = {}
            for p in station_info['pollutions']:
                aq = pd_aq[p_aq_codes[p]]
                obs_station[p] = aq
            obs[station_id] = obs_station
        
        
        #weather forecast
        pd_predict = pd.read_csv(predict_file,usecols=['test_id','PM2.5','PM10','O3'], index_col=['test_id'])
        pre={}
        for station_id in self.station_infos:
            station_info = self.station_infos[station_id]
            station_ids = []
            for i in range(forecast_hours):
                station_ids.append('{}#{}'.format(station_id,i))
            pre_station={}
            for p in station_info['pollutions']:
                pre_station[p] = pd_predict[p].ix[station_ids]
            pre[station_id] = pre_station
        
        
        #scores
        score={}
        for station_id in self.station_infos:
            station_info = self.station_infos[station_id]
            
            
            obss=[]
            for p in station_info['pollutions']:
                obss.append( obs[station_id][p].values)
            obss = np.array(obss)
            for i in range(obss.shape[1]):
                if np.isnan(obss[:,i]).any() or (obss[:,i]<0).any():
                    obss[:,i] = np.nan
            i = 0
            for p in station_info['pollutions']:
                obs[station_id][p] = obss[i]
                i+=1
            
            score_station={}
            for p in station_info['pollutions']:
                predicts = pre[station_id][p].values
                observes = obs[station_id][p]
                score_station[p] = utils.smape(observes, predicts)
                dic = {'obs':observes, 'pre':predicts}
                pd_data = pd.DataFrame(dic)
                pd_data.to_csv(os.path.join(root, predict_file_name+'{}_{}_scores.csv'.format(station_id,p)))
            score[station_id] = score_station
        
        with open(os.path.join(root, predict_file_name+'_scores.csv'), 'w') as f:
            for station_id in self.station_infos:
                station_info = self.station_infos[station_id]
                f.write(station_id)
                f.write('\n')
                for p in station_info['pollutions']:
                    f.write(p+',')
                    f.write(str(score[station_id][p])+',')
                f.write('\n')
                
        with open(os.path.join(root, predict_file_name+'_scorelist.csv'), 'w') as f:
            f.write('station_id,PM2.5,PM10,O3\n')
            for station_id in self.station_infos:
                station_info = self.station_infos[station_id]
                f.write(station_id)
                f.write(',')
                line=''
                for p in station_info['pollutions']:
                    line += (str(score[station_id][p])+',')
                line = line[:-1]
                f.write(line)
                f.write('\n')
        
        return score
  
#------------------------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    
    # prediction using model a
    #------------------------------------------------------------------------------------------------------#
    with open('./data/station_infos.json', 'r') as f:
        station_infos = json.load(f)
    with open('./data/request_paras.json', 'r') as f:
        paras = json.load(f)
    
    
    run = RunPredict(paras, station_infos,'submit_','','./predict/', use_caiyun=True,auto_submit=False)
    
    r_file = run.predict()
    '''
    utils.submit(r_file, description="")
    '''
    '''
    score = run.test_standard('./predict/submit_2018-05-29-23.csv',datetime.datetime(2018,5,31,23))
    '''
    #------------------------------------------------------------------------------------------------------#
    
   
    # prediction using model f
    #----------------------------------------------------------------------------------------------------#
    with open('./data/station_infos_f.json', 'r') as f:
        station_infos_f = json.load(f)
    with open('./data/request_paras.json', 'r') as f:
        paras = json.load(f)
    
    run_f = RunPredict(paras, station_infos_f,'submitF_','','./predict/', use_caiyun=True,auto_submit=False)
    '''
    r_file_f = run_f.predict()
    '''
    '''
    utils.submit(r_file_f, description="")
    '''
    #-----------------------------------------------------------------------------------------------------#
    
    
    
    pass










