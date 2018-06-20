# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:00:24 2018

@author: yaj
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import math
import datetime

def interpolateNan(col, maxNan):
    for i in range(len(col)):
        if np.isnan(col[i]):
            for j in range(len(col) - i):
                if not np.isnan(col[i+j]):
                    break
            if j > maxNan:
                for k in range(j):
                    col[i+k] = -1.
    
    return col

def calc_dis(x0,y0,x1,y1):
    return (x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)

'''
aq_file = './data/beijing_17_18_aq_align.csv'
meo_file = './data/beijing_17_18_meo.csv'
new_aq_file = './data/new_beijing_17_18_aq.csv'
new_meo_file = './data/new_beijing_17_18_meo.csv'

maxNan = 10
'''

'''

data_aq = pd.read_csv(aq_file)
col_PM2_5 = data_aq["PM2.5"]
col_PM2_5 = interpolateNan(col_PM2_5,maxNan)
col_PM10 = data_aq["PM10"]
col_PM10 = interpolateNan(col_PM10,maxNan)
col_O3 = data_aq["O3"]
col_O3 = interpolateNan(col_O3,maxNan)
data_aq = data_aq.interpolate()
data_aq.to_csv(new_aq_file)
'''

'''

data_meo = pd.read_csv(meo_file)
col_weather = data_meo["weather"]
col_weather = col_weather.replace("Rain", 1)
col_weather = col_weather.replace("Rain with Hail", 2)
col_weather = col_weather.replace("Rain/Snow with Hail", 3)
col_weather = col_weather.replace("Sleet", 4)
col_weather = col_weather.replace("Snow", 5)
col_weather = col_weather.replace("Sunny/clear", 6)
col_weather = col_weather.replace("Fog", 7)
col_weather = col_weather.replace("Haze", 8)
col_weather = col_weather.replace("Dust", 9)
col_weather = col_weather.replace("Sand", 10)
data_meo["weather"] = col_weather
#col_weather = interpolateNan(col_weather,maxNan)
col_temperature = data_meo["temperature"]
col_temperature = col_temperature.replace(999999,np.nan)
#col_temperature = interpolateNan(col_temperature,maxNan)
data_meo["temperature"] = col_temperature
col_pressure = data_meo["pressure"]
col_pressure = col_pressure.replace(999999,np.nan)
col_pressure = interpolateNan(col_pressure,maxNan)
data_meo["pressure"] = col_pressure
col_humidity = data_meo["humidity"]
col_humidity = col_humidity.replace(999999,np.nan)
col_humidity = interpolateNan(col_humidity,maxNan)
data_meo["humidity"] = col_humidity 
col_wind_direction = data_meo["wind_direction"]
col_wind_direction = col_wind_direction.replace(999017,-1)
col_wind_direction = col_wind_direction.replace(999999,-1)
col_wind_direction = col_wind_direction.replace(np.nan,-1)
data_meo["wind_direction"] = col_wind_direction 
col_wind_speed = data_meo["wind_speed"]
col_wind_speed = col_wind_speed.replace(np.nan,0)
col_wind_speed = col_wind_speed.replace(999999,0)
data_meo["wind_speed"] = col_wind_speed
data_meo = data_meo.interpolate()
data_meo.to_csv(new_meo_file)
'''


'''

aq_station_file = './data/beijing_aq_stations.csv'
meo_station_file = './data/beijing_meo_stations.csv'
aq_meo_station_file = './data/beijing_aq_meo_station.csv'
data_aq_station = pd.read_csv(aq_station_file)
data_meo_station = pd.read_csv(meo_station_file)
data_aq_station = data_aq_station.values
data_meo_station = data_meo_station.values
aq_meo_station = []
for i in range(len(data_aq_station)):
    dis = calc_dis(data_aq_station[i][1],data_aq_station[i][2],data_meo_station[0][1],data_meo_station[0][2])
    index = 0
    for j in range(len(data_meo_station)):
        dis2 = calc_dis(data_aq_station[i][1],data_aq_station[i][2],data_meo_station[j][1],data_meo_station[j][2])
        if (dis2<dis):
            dis = dis2
            index = j
    aq_meo_station.append([data_aq_station[i][0], data_meo_station[index][0]])
aq_meo_station_df = pd.DataFrame(aq_meo_station, columns=['aq_station','meo_station'])
aq_meo_station_df.to_csv(aq_meo_station_file)
'''


'''

aq_station_file = './data/beijing_aq_stations.csv'
meo_grid_file = './data/Beijing_historical_meo_grid.csv'
x_s=115
x_e=118
y_s=39
y_e=41
dx=0.1
dy=0.1
one_row = math.floor((y_e-y_s)/dy)+1
data_aq_station = pd.read_csv(aq_station_file)
data_meo_grid = pd.read_csv(meo_grid_file)
array_aq_station = data_aq_station.values
station_grid=[]
for i in range(len(array_aq_station)):
    x = array_aq_station[i][2]
    y = array_aq_station[i][3]
    rows = math.floor((x - x_s)/dx)
    cols = math.floor((y - y_s)/dy)
    grids = rows*one_row+cols
    station_grid.append('beijing_grid_{:0=3d}'.format(grids))
    grid_code = 'beijing_grid_{:0=3d}'.format(grids)
    station_data = data_meo_grid.loc[data_meo_grid['stationName'] == grid_code]
    col_wind_direction = station_data['wind_direction']
    col_wind_direction = col_wind_direction.replace(999017,-1)
    station_data['wind_direction'] = col_wind_direction
    station_data.to_csv('./data/from_grid/from_grid_beijing_{}.csv'.format(array_aq_station[i][1]))
data_aq_station['grid'] = pd.Series(station_grid)
data_aq_station.to_csv(aq_station_file)
'''

'''

new_aq_file = './data/new_beijing_17_18_aq.csv'
aq_station_file = './data/beijing_aq_stations.csv'
data_aq_station = pd.read_csv(aq_station_file, usecols=[1,4])
aq_stations = data_aq_station.values
data_aq = pd.read_csv(new_aq_file, usecols=[1,2,3,4,5])
root = './data/from_aq/'
for i in range(len(aq_stations)):
    station_code = aq_stations[i][0]
    station_data = data_aq.loc[data_aq['stationId'] == station_code]
    file = os.path.join(root, 'from_aq_beijing_{}.csv'.format(station_code))
    station_data.to_csv(file)
'''


########################################################################################################################
def calc_station_grid(station_file, x_s, x_e, y_s, y_e, dx, dy, prefix):
    '''
    find the gird based on the lon&lat of the stations
    '''
    one_row = math.floor((y_e-y_s)/dy)+1
    data_station = pd.read_csv(station_file, usecols=['station_id','longitude','latitude'])
    array_station = data_station.values
    station_grid=[]
    for i in range(len(array_station)):
        x = array_station[i][1]
        y = array_station[i][2]
        rows = math.floor((x - x_s)/dx)
        cols = math.floor((y - y_s)/dy)
        grids = rows*one_row+cols
        station_grid.append(prefix + '{:0=3d}'.format(grids))
    data_station['grid'] = pd.Series(station_grid)
    data_station.to_csv(station_file)


#calc_station_grid('./data/london_aq_stations.csv', -2., 2., 50.5, 52.5, 0.1, 0.1, 'london_grid_')
#########################################################################################################################


################################################################################################################################
def get_data_from_grid(station_file, grid_file, folder, prefix):
    '''
    data from the meo grid file
    '''
    start = datetime.datetime(2017,1,1,0)
    end = datetime.datetime(2018,1,30,23)
    
    data_station = pd.read_csv(station_file, usecols=['station_id','grid'])
    data_grid = pd.read_csv(grid_file)
    array_station = data_station.values
    for i in range(len(array_station)):
        grid_code = array_station[i][1]
        station_data = data_grid.loc[data_grid['stationName'] == grid_code]
        col_wind_direction = station_data['wind_direction']
        col_wind_direction = col_wind_direction.replace(999017,-1)
        station_data['wind_direction'] = col_wind_direction
        ts = pd.DataFrame(np.zeros((9480,1)),
                          index = pd.date_range(start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M'),freq='h'),
                          columns=['weather'])
        station_data = station_data+ts
        station_data = station_data[~station_data.index.duplicated()]
        station_data = station_data.fillna(-1)
        station_data.to_csv(os.path.join(folder, prefix+'{}.csv'.format(array_station[i][0])))
        
    return

#get_data_from_grid('./data/london_aq_stations.csv', './data/London_historical_meo_grid.csv', './data/from_grid/', 'from_grid_london_')
####################################################################################################################################


##############################################################################################################################
def get_data_from_aqData(station_file, aq_file, new_aq_file, folder, prefix, inter_cols, maxNan=10, fille_nan = -1):
    '''
    air quality data
    '''
    start = datetime.datetime(2017,1,1,0)
    end = datetime.datetime(2018,1,30,23)
    
    data_station = pd.read_csv(station_file, usecols=['station_id'])
    data_aq = pd.read_csv(aq_file)
    array_station = data_station.values
#    for col in inter_cols:
#        col_data = data_aq[col]
#        col_data = interpolateNan(col_data,maxNan)
#        data_aq[col] = col_data
#    data_aq = data_aq.interpolate()
#    data_aq.to_csv(new_aq_file)
    for i in range(len(array_station)):
        station_code = array_station[i][0]
        station_data = data_aq.loc[data_aq['stationId'] == station_code]
        ts = pd.DataFrame(np.zeros((9480,1)),
                          index = pd.date_range(start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M'),freq='h'),
                          columns=['weather'])
        station_data = station_data+ts
        station_data = station_data[~station_data.index.duplicated()]
        for col in inter_cols:
            col_data = station_data[col]
            col_data = interpolateNan(col_data,maxNan)
            station_data[col] = col_data
        station_data = station_data.interpolate()
        file = os.path.join(folder, prefix+'{}.csv'.format(array_station[i][0]))
        station_data.to_csv(file)

#get_data_from_aqData('./data/london_aq_stations.csv', './data/London_historical_aqi_forecast_stations_20180331_align.csv', 
#                    './data/new_London_historical_aqi_forecast_stations_20180331.csv', './data/from_aq/', 'from_aq_london_',
#                    inter_cols=['PM2.5','PM10'])
#################################################################################################################################


##############################################################################################################################
def get_waether_from_meo(station_file,meo_file,new_meo_file, folder, prefix,):
    '''
    weather description data
    '''
    start = datetime.datetime(2017,1,1,0)
    end = datetime.datetime(2018,1,30,23)
    
    data_station = pd.read_csv(station_file, usecols=['aq_station','meo_station'])
    array_station = data_station.values
    data_meo = pd.read_csv(meo_file, usecols=['station_id','weather','utc_time'], index_col=['utc_time'])
    col_weather = data_meo["weather"]
#    col_weather = col_weather.replace("Rain", 1)
#    col_weather = col_weather.replace("Rain with Hail", 2)
#    col_weather = col_weather.replace("Rain/Snow with Hail", 3)
#    col_weather = col_weather.replace("Sleet", 4)
#    col_weather = col_weather.replace("Snow", 5)
#    col_weather = col_weather.replace("Sunny/clear", 6)
#    col_weather = col_weather.replace("Fog", 7)
#    col_weather = col_weather.replace("Haze", 8)
#    col_weather = col_weather.replace("Dust", 9)
#    col_weather = col_weather.replace("Sand", 10)
    col_weather = col_weather.replace("Rain", 1)
    col_weather = col_weather.replace("Rain with Hail", 1)
    col_weather = col_weather.replace("Rain/Snow with Hail", 1)
    col_weather = col_weather.replace("Dust", 3)
    col_weather = col_weather.replace("Sand", 3)
    col_weather = col_weather.replace("\S+", 2, regex=True)
    data_meo["weather"] = col_weather
    data_meo.to_csv(new_meo_file)
    for i in range(len(array_station)):
        station_code = array_station[i][0]
        meo_code = array_station[i][1]
        station_data = data_meo.loc[data_meo['station_id'] == meo_code]
        #station_data = station_data.ix[start.strftime('%Y-%m-%d %H:%M'):end.strftime('%Y-%m-%d %H:%M')]
        ts = pd.DataFrame(np.zeros((9480,1)),
                          index = pd.date_range(start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M'),freq='h'),
                          columns=['weather'])
        station_data = station_data+ts
        station_data = station_data[~station_data.index.duplicated()]
        station_data = station_data.fillna(-1)
        file = os.path.join(folder, prefix+'{}.csv'.format(station_code))
        station_data.to_csv(file,columns=['weather'])

#get_waether_from_meo('./data/beijing_aq_meo_station.csv', './data/beijing_17_18_meo.csv','./data/beijing_17_18_meo_new.csv','./data/from_meo/','from_meo_beijing_')
######################################################################################################################



######################################################################################################################################
def get_no2_from_aqData(station_file, aq_file, folder, prefix,start,end):
    '''
    NO2 data
    '''
    total_hours = ((end-start).days+1)*24
    
    data_station = pd.read_csv(station_file, usecols=['aq_station','meo_station'])
    array_station = data_station.values
    
    data_aq = pd.read_csv(aq_file, usecols=['stationId','utc_time','NO2'], 
                          parse_dates =['utc_time'] ,index_col=['utc_time'])
    
    for i in range(len(array_station)):
        station_code = array_station[i][0]
        station_data = data_aq.loc[data_aq['stationId'] == station_code]
        ts = pd.DataFrame(np.zeros((total_hours,1)),
                          index = pd.date_range(start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M'),freq='h'),
                          columns=['NO2'])
        station_data = station_data+ts
        station_data = station_data[~station_data.index.duplicated()]
        station_data = station_data.interpolate(limits=10)
        station_data = station_data.fillna(-1)
        file = os.path.join(folder, prefix+'{}.csv'.format(station_code))
        station_data.to_csv(file,columns=['NO2'])
    
    pass

#get_no2_from_aqData('./data/beijing_aq_meo_station.csv', './data/beijing_17_18_aq.csv', './data/from_no2/','no2_beijing_',
#                    datetime.datetime(2017,1,1),datetime.datetime(2018,1,30,23))
#####################################################################################################################################

















