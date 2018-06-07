# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:18:02 2018

@author: Cookie-2
"""

import os
import requests
import pandas as pd
import pytz
import datetime
import urllib

#Air Quality data
def request_aq(city, start_time, end_time, save_file):
    url = 'https://biendata.com/competition/airquality/{}/{}/{}/2k0d1d8'.format(city, start_time, end_time)
    print(url)
    respones= requests.get(url)
    with open (save_file,'w') as f:
        f.write(respones.text)

#Observed Meteorology Data
def request_meo(city, start_time, end_time, save_file):
    url = 'https://biendata.com/competition/meteorology/{}/{}/{}/2k0d1d8'.format(city, start_time, end_time)
    print(url)
    respones= requests.get(url)
    with open (save_file,'w') as f:
        f.write(respones.text)

#Grid Meteorology Data.
def request_meo_grid(city, start_time, end_time, save_file):
    url = 'https://biendata.com/competition/meteorology/{}_grid/{}/{}/2k0d1d8'.format(city, start_time, end_time)
    print(url)
    respones= requests.get(url)
    #print(save_file)
    with open (save_file,'w') as f:
        f.write(respones.text)

def request_meo_grid_station(stations, city, start_time, end_time, save_file, root):
    data_aq_station = pd.read_csv(stations, usecols=[1,4])
    aq_stations = data_aq_station.values
    request_meo_grid(city, start_time, end_time, save_file)
    data_grid = pd.read_csv(save_file)
    for i in range(len(aq_stations)):
        station_data = data_grid.loc[data_grid['station_id'] == aq_stations[i][0]]
        col_wind_direction = station_data['wind_direction']
        col_wind_direction = col_wind_direction.replace(999017,-1)
        station_data['wind_direction'] = col_wind_direction
        station_data.to_csv(os.path.join(root, '/from_aq_{}_{}.csv'.format(city, aq_stations[i][0])))

#获取当日各种数据
def request_today(city,folder,staions_file,backward_days):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    tz = pytz.timezone('utc')
    now_date = datetime.datetime.now(tz).strftime("%Y-%m-%d")
    now_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H-%M")
    folder = os.path.join(folder, now_date)
    folder_grid = os.path.join(folder, '/grids/')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    if not os.path.isdir(folder_grid):
        os.mkdir(folder_grid)
    
    now = datetime.datetime.now(tz)
    end_time = now.strftime("%Y-%m-%d-%H")
    start = now - datetime.timedelta(hours=backward_days*24)+1
    start_time = start.strftime("%Y-%m-%d-%H")
    
    request_aq(city, start_time, end_time, os.path.join(folder, city+'_aq_'+now_time+'.csv'))
    print('*'*5)
    request_meo(city, start_time, end_time, os.path.join(folder, city+'_meo_'+now_time+'.csv'))
    print('*'*5)
    request_meo_grid_station(staions_file,city, start_time, end_time, os.path.join(folder, city+'_grid_'+now_time+'.csv'),folder_grid)


def request_real_data(cities, backward_hours, folder, now=None):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if now == None:
        tz = pytz.timezone('utc')
        now = datetime.datetime.now(tz)
    #now_time = now.strftime("%Y-%m-%d %H-%M")
    folder_aq = os.path.join(folder, 'bien', 'aq')
    folder_grid = os.path.join(folder, 'bien', 'grid')
    if not os.path.isdir(folder_aq):
        os.makedirs(folder_aq)
    if not os.path.isdir(folder_grid):
        os.makedirs(folder_grid)
    
    end_time = now.strftime("%Y-%m-%d-%H")
    start = now - datetime.timedelta(hours=backward_hours-1)
    start_time = start.strftime("%Y-%m-%d-%H")
    
    for city in cities:
        request_aq(city, start_time, end_time, os.path.join(folder_aq, city+'_aq'+'.csv'))
        #print('*'*5)
        #request_meo(city, start_time, end_time, os.path.join(folder_aq, city+'_meo_'+'.csv'))
        #print('*'*5)
        request_meo_grid(city, start_time, end_time, os.path.join(folder_grid, city+'_grid'+'.csv'))


def request_real_data_aq(cities, backward_hours, folder, now=None):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if now == None:
        tz = pytz.timezone('utc')
        now = datetime.datetime.now(tz)
    #now_time = now.strftime("%Y-%m-%d %H-%M")
    folder_aq = os.path.join(folder, 'bien', 'aq')
    if not os.path.isdir(folder_aq):
        os.makedirs(folder_aq)
    
    end_time = now.strftime("%Y-%m-%d-%H")
    start = now - datetime.timedelta(hours=backward_hours-1)
    start_time = start.strftime("%Y-%m-%d-%H")
    
    for city in cities:
        request_aq(city, start_time, end_time, os.path.join(folder_aq, city+'_aq'+'.csv'))














