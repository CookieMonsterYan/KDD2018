# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 20:04:46 2018

@author: Cookie-2
"""


import numpy as np
import json
import os
import pytz
import datetime
import requests
import shutil
import pandas as pd

from real_time_data_api import request_real_data
from predicted_meo_data_api import request_from_openweather, request_from_accuweather
import predicted_meo_data_api


def smape(actual, predicted):
    predicted = predicted[~np.isnan(actual)]
    actual = actual[~np.isnan(actual)]    
    
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)
     
    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe'))



class normalization(object):
    def __init__(self, typeNo=None, means=None, stds=None, maxs=None, mins=None):
        self.means = means
        self.stds = stds
        self.maxs = maxs
        self.mins = mins
        self.typeNo = np.array(typeNo)
            
    def changeType(self, typeNo):
        self.typeNo = typeNo
        
    def __call__(self, data, forward=True):
        if forward:
            if self.typeNo == 1:
                data -= self.means
                data /= self.stds
                return data
            elif self.typeNo == 2:
                data -= self.mins
                data /= (self.maxs - self.mins)
                return data
            
            else:
                return data
        else:
            if self.typeNo == 1:
                data *= self.stds
                data += self.means                
                return data
            elif self.typeNo == 2:
                data *= (self.maxs - self.mins)
                data += self.mins                
                return data
            else:
                return data
        
    def save(self, file):
        
        data = {
                'typeNo':self.typeNo.tolist(),
                'means':self.means.tolist(),
                'stds':self.stds.tolist(),
                'maxs':self.maxs.tolist(),
                'mins':self.mins.tolist()
                }
        with open(file, 'w') as f:
            json.dump(data,f)
    
    def load(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
        self.typeNo = np.array(data['typeNo'])
        self.means = np.array(data['means'])
        self.stds = np.array(data['stds'])
        self.maxs = np.array(data['maxs'])
        self.mins = np.array(data['mins'])


def check_data_files(folder, stations, use_caiyun):
    '''
    check if all data needed has been downloaded
    '''
    for station in stations:
        
        if use_caiyun:
            file_caiyun = os.path.join(folder, 'caiyun', station['station_id']+'.csv')
            if not os.path.isfile(file_caiyun):
                return False
        else:
            file_openw = os.path.join(folder, 'openw', station['station_id']+'.csv')
            if not os.path.isfile(file_openw):
                return False
            file_accu = os.path.join(folder, 'accu', station['station_id']+'.csv')
            if not os.path.isfile(file_accu):
                return False
        
        file_aq = os.path.join(folder, 'bien', 'aq', station['station_id']+'.csv')
        if not os.path.isfile(file_aq):
            return False
        file_grid = os.path.join(folder, 'bien', 'grid', station['station_id']+'.csv')
        if not os.path.isfile(file_grid):
            return False
    
    return True
        
def clip_data(folder, stations, now, backward_hours, forecast_hours, 
              interpolate=True, delete_duplicated = True, clean_minus_aq = True,
              fill_nan_aq_file = True,
              use_caiyun=False):
    '''
    split the grid data into data files for each station
    '''
    end = now
    start = now - datetime.timedelta(hours=backward_hours-1)
    predict_start = now + datetime.timedelta(hours=1)
    predict_end = now + datetime.timedelta(hours=forecast_hours)
    for station in stations:
        if use_caiyun:
            #caiyun
            clip_caiyun_data(folder, stations, now, forecast_hours, interpolate=True)
        else:
            #openw
            the_folder = os.path.join(folder, 'openw')
            file = os.path.join(the_folder, station['city'] + '_ok' +'.csv')
            sta_file = os.path.join(the_folder, station['station_id']+'.csv')
            if os.path.isfile(file):
                shutil.copy(file, sta_file)
            else:
                file = os.path.join(the_folder, station['city']+'.csv')
                #print(file)
                if not os.path.isfile(file):
                    return -1
                data = pd.read_csv(file,usecols=['utc','pressure'],parse_dates=['utc'],index_col=['utc'])
                new_data = data[predict_start.strftime('%Y-%m-%d %H:%M'):predict_end.strftime('%Y-%m-%d %H:%M')]
                ts = pd.DataFrame(pd.Series(np.zeros(forecast_hours),
                                            pd.date_range(predict_start.strftime('%Y-%m-%d %H:%M'), periods=forecast_hours,freq='h')),
                        columns=['pressure'])
                new_data = new_data + ts
                if interpolate:
                    new_data = new_data.interpolate(limit_direction='both',limit_area='both')
                file = os.path.join(the_folder, station['city'] + '_ok' +'.csv')
                sta_file = os.path.join(the_folder, station['station_id']+'.csv')
                #new_data.interpolate()
                new_data.to_csv(file)
                shutil.copy(file, sta_file)
            #accu
            the_folder = os.path.join(folder, 'accu')
            sta_file = os.path.join(the_folder, station['station_id']+'.csv')
            if not os.path.isfile(sta_file):
                return -2
        
        #bien_aq
        the_folder = os.path.join(folder, 'bien','aq')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        file = os.path.join(the_folder, station['bien_code'] + '_aq' +'.csv')
        if not os.path.isfile(file):
            return -3
        data = pd.read_csv(file, usecols=['station_id','time','PM25_Concentration','PM10_Concentration','O3_Concentration','NO2_Concentration'], 
                           parse_dates=['time'],index_col=['station_id','time'])
        #print(station['station_id'])
        new_data = data.ix[station['station_id']]
        #print(new_data.head(3))
        new_data = new_data.ix[start.strftime('%Y-%m-%d %H:%M'):end.strftime('%Y-%m-%d %H:%M')]
        ts = pd.DataFrame(np.zeros((backward_hours,4)),
                          index = pd.date_range(start.strftime('%Y-%m-%d %H:%M'), periods=backward_hours,freq='h'),
                          columns=['PM25_Concentration','PM10_Concentration','O3_Concentration','NO2_Concentration'])
        new_data = new_data + ts
        
        #2018/04/26 
        if delete_duplicated:
            if len(new_data.index) > backward_hours:
                new_data = new_data[~new_data.index.duplicated()]
        #-------------------------------------
        
        #2018/05/02 
        if clean_minus_aq:
            new_data[new_data<0.]=np.nan
            pass            
        #--------------------------------------
        
        if interpolate:
            new_data = new_data.interpolate(limit_direction='both',limit_area='both')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        new_data.to_csv(sta_file)
        
        #bien_grid
        the_folder = os.path.join(folder, 'bien','grid')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        file = os.path.join(the_folder, station['bien_code'] + '_grid' +'.csv')
        if not os.path.isfile(file):
            return -4
        data = pd.read_csv(file, usecols=['station_id','time','temperature','pressure','humidity','wind_direction','wind_speed','weather'], 
                           parse_dates=['time'],index_col=['station_id','time'])
        new_data = data.ix[station['grid']]
        new_data = new_data.ix[start.strftime('%Y-%m-%d %H:%M'):end.strftime('%Y-%m-%d %H:%M')] 
        col_weather = new_data['weather']
        col_weather = col_weather.replace("RAIN", 1)
        col_weather = col_weather.replace("RAIN.*", 1, regex=True)
        col_weather = col_weather.replace("DUST", 3)
        col_weather = col_weather.replace("SAND", 3)
        col_weather = col_weather.replace("\S+", 2, regex=True)
        new_data['weather'] = col_weather
        ts = pd.DataFrame(np.zeros((backward_hours,6)),
                          index=pd.date_range(start.strftime('%Y-%m-%d %H:%M'), periods=backward_hours,freq='h'),
                          columns=['temperature','pressure','humidity','wind_direction','wind_speed','weather'])
        new_data = new_data + ts
        
        #2018/05/23 
        if delete_duplicated:
            if len(new_data.index) > backward_hours:
                new_data = new_data[~new_data.index.duplicated()]
        #-------------------------------------
        
        if interpolate:
            new_data = new_data.interpolate(limit_direction='both',limit_area='both')
        new_data.rename(columns={'wind_speed':'wind_speed/kph'},inplace=True)
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        new_data.to_csv(sta_file)
    
    if fill_nan_aq_file:
        fill_na_aq_data(folder,stations)    
    
    return 0


def clip_real_data(folder, stations, now, backward_hours, interpolate=True, delete_duplicated = True):
    '''
    split the grid data into data files for each station, only live data
    '''
    end = now
    start = now - datetime.timedelta(hours=backward_hours-1)
    for station in stations:
        #bien_aq
        the_folder = os.path.join(folder, 'bien','aq')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        file = os.path.join(the_folder, station['bien_code'] + '_aq' +'.csv')
        if not os.path.isfile(file):
            return -3
        data = pd.read_csv(file, usecols=['station_id','time','PM25_Concentration','PM10_Concentration','O3_Concentration'], 
                           parse_dates=['time'],index_col=['station_id','time'])
        #print(station['station_id'])
        new_data = data.ix[station['station_id']]
        #print(new_data.head(3))
        new_data = new_data.ix[start.strftime('%Y-%m-%d %H:%M'):end.strftime('%Y-%m-%d %H:%M')]
        ts = pd.DataFrame(np.zeros((backward_hours,3)),
                          index = pd.date_range(start.strftime('%Y-%m-%d %H:%M'), periods=backward_hours,freq='h'),
                          columns=['PM25_Concentration','PM10_Concentration','O3_Concentration'])
        new_data = new_data + ts
        
        #2018/04/26 
        if delete_duplicated:
            if len(new_data.index) > backward_hours:
                new_data = new_data[~new_data.index.duplicated()]
        #-------------------------------------
        
        if interpolate:
            new_data = new_data.interpolate(limit_direction='both',limit_area='both')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        new_data.to_csv(sta_file)
        
        #bien_grid
        the_folder = os.path.join(folder, 'bien','grid')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        file = os.path.join(the_folder, station['bien_code'] + '_grid' +'.csv')
        if not os.path.isfile(file):
            return -4
        data = pd.read_csv(file, usecols=['station_id','time','temperature','pressure','humidity','wind_direction','wind_speed'], 
                           parse_dates=['time'],index_col=['station_id','time'])
        new_data = data.ix[station['grid']]
        new_data = new_data.ix[start.strftime('%Y-%m-%d %H:%M'):end.strftime('%Y-%m-%d %H:%M')] 
        ts = pd.DataFrame(np.zeros((backward_hours,5)),
                          index=pd.date_range(start.strftime('%Y-%m-%d %H:%M'), periods=backward_hours,freq='h'),
                          columns=['temperature','pressure','humidity','wind_direction','wind_speed'])
        new_data = new_data + ts
        if interpolate:
            new_data = new_data.interpolate(limit_direction='both',limit_area='both')
        new_data.rename(columns={'wind_speed':'wind_speed/kph'},inplace=True)
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        new_data.to_csv(sta_file)

def clip_real_data_aq(folder, stations, now, backward_hours, interpolate=True, delete_duplicated = True):
    end = now
    start = now - datetime.timedelta(hours=backward_hours-1)
    for station in stations:
        #bien_aq
        the_folder = os.path.join(folder, 'bien','aq')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        file = os.path.join(the_folder, station['bien_code'] + '_aq' +'.csv')
        if not os.path.isfile(file):
            return -3
        data = pd.read_csv(file, usecols=['station_id','time','PM25_Concentration','PM10_Concentration','O3_Concentration'], 
                           parse_dates=['time'],index_col=['station_id','time'])
        #print(station['station_id'])
        new_data = data.ix[station['station_id']]
        #print(new_data.head(3))
        new_data = new_data.ix[start.strftime('%Y-%m-%d %H:%M'):end.strftime('%Y-%m-%d %H:%M')]
        ts = pd.DataFrame(np.zeros((backward_hours,3)),
                          index = pd.date_range(start.strftime('%Y-%m-%d %H:%M'), periods=backward_hours,freq='h'),
                          columns=['PM25_Concentration','PM10_Concentration','O3_Concentration'])
        new_data = new_data + ts
        
        #2018/04/26
        if delete_duplicated:
            if len(new_data.index) > backward_hours:
                new_data = new_data[~new_data.index.duplicated()]
        #-------------------------------------
        
        if interpolate:
            new_data = new_data.interpolate(limit_direction='both',limit_area='both')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        new_data.to_csv(sta_file)
        
def clip_caiyun_data(folder, stations, now, predict_hours, interpolate=True):
    '''
    split the caiyun grid data into data files for each station
    '''
    start = now + datetime.timedelta(hours=1)
    end = now + datetime.timedelta(hours=predict_hours)
    for station in stations:
        the_folder = os.path.join(folder, 'caiyun')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        file = os.path.join(the_folder, station['bien_code'] + '_grid' +'.csv')
        if not os.path.isfile(file):
            return -4
        data = pd.read_csv(file, usecols=['station_id','forecast_time','temperature','pressure','humidity','wind_direction','wind_speed','weather'], 
                           parse_dates=['forecast_time'],index_col=['station_id','forecast_time'])
        new_data = data.ix[station['grid']]
        new_data = new_data.ix[start.strftime('%Y-%m-%d %H:%M'):end.strftime('%Y-%m-%d %H:%M')] 
        col_weather = new_data['weather']
        col_weather = col_weather.replace("RAIN", 1)
        col_weather = col_weather.replace("RAIN.*", 1, regex=True)
        col_weather = col_weather.replace("DUST", 3)
        col_weather = col_weather.replace("SAND", 3)
        col_weather = col_weather.replace("\S+", 2, regex=True)
        new_data['weather'] = col_weather
        ts = pd.DataFrame(np.zeros((predict_hours,6)),
                          index=pd.date_range(start.strftime('%Y-%m-%d %H:%M'), periods=predict_hours,freq='h'),
                          columns=['temperature','pressure','humidity','wind_direction','wind_speed','weather'])
        new_data = new_data + ts
        if interpolate:
            new_data = new_data.interpolate(limit_direction='both',limit_area='both')
        new_data.rename(columns={'wind_speed':'wind_speed/kph'},inplace=True)
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        new_data.to_csv(sta_file)
    
    pass


def fill_na_aq_data(folder, stations):
    '''
    if the air quality data in a station is all null, use the data in the nearest station to fill
    '''
    air_code = {'Beijing':['PM25_Concentration','PM10_Concentration','O3_Concentration','NO2_Concentration'],
                'London':['PM25_Concentration','PM10_Concentration']}
    
    grid_nos=[]
    for i in range(len(stations)):
        stations[i]['grid_no'] = int(stations[i]['grid'].split('_')[-1])
        grid_nos.append(stations[i]['grid_no'])
    grid_nos = np.array(grid_nos)
    
    for station in stations:
        the_folder = os.path.join(folder, 'bien','aq')
        sta_file = os.path.join(the_folder, station['station_id']+'.csv')
        
        data = pd.read_csv(sta_file)
        for air in air_code[station['city']]:
            if data[air].isnull().any():
                print('{}:{}is null'.format(sta_file, air))
                this_grid = station['grid_no']
                delt_grids = np.abs(grid_nos - this_grid)
                indexs = np.argsort(delt_grids)
                for i in indexs:
                    if station['city'] != stations[i]['city']:
                        continue
                    source_file = os.path.join(the_folder, stations[i]['station_id']+'.csv')
                    source_data = pd.read_csv(source_file,usecols=[air])
                    if not source_data[air].isnull().any():
                        data[air] = source_data[air]
                        break
                data.to_csv(sta_file,index=False)
                    



def request_data(paras, now=None, use_caiyun=False):
    '''
    request all data needed to do prediction
    '''
    root = paras['root']
    backward_hours = paras['backward']
    forecast_hours = paras['forecast']
    
    if now == None:
        tz = pytz.timezone('utc')
        now = datetime.datetime.now(tz)
    now = datetime.datetime(now.year,now.month,now.day,now.hour)
    now_time = now.strftime("%Y-%m-%d-%H")
    folder = os.path.join(root, now_time)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
    if check_data_files(folder, paras['stations'], use_caiyun):
        print('all data needed already in {}'.format(folder))
        return folder
    
    
    if use_caiyun:
        #caiyun: predicterd meo
        caiyun = paras['bien']
        predicted_meo_data_api.request_from_caiyun(caiyun['cities'], now, folder)
    else:
        #openweather: predicted meo
        openw = paras['openw']
        request_from_openweather(openw['cities'],openw['country_codes'],folder)
        #accu weather: predicted meo
        accu = paras['accu']
        request_from_accuweather(accu['stations'],forecast_hours,folder,now, check_requested=True)
    
    #biendata:real_time data
    bien = paras['bien']
    request_real_data(bien['cities'], backward_hours, folder, now)
    
    
    pre_ok = clip_data(folder, paras['stations'], now, backward_hours, forecast_hours, use_caiyun=use_caiyun)
    
       
    if pre_ok != 0:
        print(pre_ok)
        return
    
    return folder


def prepare_data(data_folder, station_id, method_type, norm, pollution, use_caiyun):
    '''
    prepare the x data for prediction
    '''
    if method_type == 'model_a' or method_type == 'model_f':
        meo_codes = ['temperature', 'pressure',	'humidity', 'wind_direction',	'wind_speed/kph']
        p_aq_codes = {'PM2.5':'PM25_Concentration','PM10':'PM10_Concentration','O3':'O3_Concentration'}
        aq_file = os.path.join(data_folder,'bien','aq',station_id+'.csv')
        grid_file = os.path.join(data_folder,'bien','grid',station_id+'.csv')
        openw_file = os.path.join(data_folder,'openw',station_id+'.csv')
        accu_file = os.path.join(data_folder,'accu',station_id+'.csv')
        caiyun_file = os.path.join(data_folder,'caiyun',station_id+'.csv')
        
        data_aq = pd.read_csv(aq_file)
        data_grid = pd.read_csv(grid_file)
        if use_caiyun:
            data_caiyun = pd.read_csv(caiyun_file)
            data_pre = data_caiyun
        else:
            data_openw = pd.read_csv(openw_file)
            data_accu = pd.read_csv(accu_file)
            data_pre = pd.concat([data_accu, data_openw],axis=1)
        
        x_aq = data_aq[p_aq_codes[pollution]]
        x_meo_obs = data_grid[meo_codes]
        x_pre = data_pre[meo_codes]
        
        x_aq = x_aq.values
        x_aq = x_aq.astype('float32')
        x_meo_obs = x_meo_obs.values
        x_meo_obs = x_meo_obs.astype('float32')
        x_pre = x_pre.values
        x_pre = x_pre.astype('float32')
        
        x_aq = x_aq.reshape((-1,1))
        x_obs=np.concatenate((x_aq,x_meo_obs), axis=1)
        
        x_obs = x_obs.reshape((1,x_obs.shape[0],x_obs.shape[1]))
        x_pre = x_pre.reshape((1,x_pre.shape[0],x_pre.shape[1]))
        
        norm_x_obs = normalization()
        norm_x_obs.load(norm['norm_x_obs'])
        norm_x_obs(x_obs)
        
        norm_x_pre = normalization()
        norm_x_pre.load(norm['norm_x_pre'])
        norm_x_pre(x_pre)
        
        #print(x_obs.shape)
        #print(x_pre.shape)
        
        return [x_obs,x_pre]
       
    else:
        return None
    

def submit(file, description="submit"):
    print(file)
    
    files={'files': open(file,'rb')}
    
    data = {
        "user_id": "frozencookie",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
        "team_token": "35aa5090c41b258a14d8261e654d6abd660206417a91eba10eb4fa9cdae831d6", #your team_token.
        "description": description,  #no more than 40 chars.
        "filename": os.path.split(file)[1], #your filename
    }
    
    url = 'https://biendata.com/competition/kdd_2018_submit/'
    
    response = requests.post(url, files=files, data=data)
    
    print(response.text)



def change_station_info(old_info,new_info,use_model_file):
    aq_code = ['PM2.5','PM10','O3']
    with open(old_info, 'r') as f:
        station_infos = json.load(f)
    use_models = pd.read_csv(use_model_file, index_col=['station_id'])
    for station in use_models.index:
        city = station_infos[station]['city']
        if city == 'Beijing':
            for aq in aq_code:
                model = use_models[aq].loc[station]
                station_infos[station]['model_file'][aq] = os.path.join('./model/',model,city,aq,model+'_'+station+'.h5')
                station_infos[station]['model_id'][aq] = model
                station_infos[station]['norm'][aq]['norm_x_obs'] = os.path.join('./model/',model,city,aq,'norm_x_obs.json')
                station_infos[station]['norm'][aq]['norm_x_pre'] = os.path.join('./model/',model,city,aq,'norm_x_pre.json')
                station_infos[station]['norm'][aq]['norm_y'] = os.path.join('./model/',model,city,aq,'norm_y.json')
        else:
            for aq in aq_code[:-1]:
                model = use_models[aq].loc[station]
                station_infos[station]['model_file'][aq] = os.path.join('./model/',model,city,aq,model+'_'+station+'.h5')
                station_infos[station]['model_id'][aq] = model
                station_infos[station]['norm'][aq]['norm_x_obs'] = os.path.join('./model/',model,city,aq,'norm_x_obs.json')
                station_infos[station]['norm'][aq]['norm_x_pre'] = os.path.join('./model/',model,city,aq,'norm_x_pre.json')
                station_infos[station]['norm'][aq]['norm_y'] = os.path.join('./model/',model,city,aq,'norm_y.json')
                
    with open(new_info, 'w') as f:
        json.dump(station_infos,f)











