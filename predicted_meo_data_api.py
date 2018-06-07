# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:32:01 2018

@author: yaj
"""


import os
import sys
import urllib.request
import json
import csv
import time
import pytz
import datetime
from bs4 import BeautifulSoup
import math
import requests

import pandas as pd
import numpy as np


def request_from_openweather(city_names, country_codes,save_file_folder = './data/daily/'):
    if not os.path.isdir(save_file_folder):
        os.makedirs(save_file_folder)
    
    folder = os.path.join(save_file_folder, 'openw')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    open_weather_api_key = ''
    
    for city_name in city_names:
        url = 'http://api.openweathermap.org/data/2.5/forecast?q={},{}&appid={}'.format(
                city_name,country_codes[city_name],open_weather_api_key)
        print(url)
        
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        rescode = response.getcode()
        
        if(rescode==200):
            response_body = response.read()
            result = response_body.decode('utf-8')
            raw = json.loads(result)
            
            json_file = os.path.join(folder, city_name+'.json')
            with open(json_file, 'w') as f:
                json.dump(raw, f)
                
            dlist = raw['list']
            csv_file = os.path.join(folder, city_name+'.csv')
            with open(csv_file, 'w') as f:
                f.write(',utc,temperature,pressure,humidity,wind_direction,wind_speed/kph\n')
                for i in range(len(dlist)):
                    f.write(str(i)+',')
                    d = dlist[i]
                    f.write(d['dt_txt']+',')
                    f.write(str(d['main']['temp']-273.15)+',')
                    f.write(str(d['main']['pressure'])+',')
                    f.write(str(d['main']['humidity'])+',')
                    f.write(str(d['wind']['deg'])+',')
                    f.write(str(d['wind']['speed']*3.6))
                    f.write('\n')        
        else:
            print("Error Code:" + rescode)
            return


def request_from_accuweather(stations, predict_hours, save_file_folder, now=None, check_requested=True):
    if now == None:
        tz = pytz.timezone('utc')
        now = datetime.datetime.now(tz)
    now_hour = now.hour
    #print(save_file_folder)
    folder = os.path.join(save_file_folder, 'accu')
    #print(folder)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
    headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.96 Safari/537.36'
            }
    winddirection = {
            "":0,
            "N":0,"NNE":22.5,"NE":45,"ENE":67.5,
            "E":90,"ESE":112.5,"SE":135,"SSE":157.5,
            "S":180,"SSW":202.5,"SW":225.,"WSW":247.5,
            "W":270,"WNW":292.5,"NW":315,"NNW":337.5
            }
    
    hours_step = 8
    
    for i in range(len(stations)):
        station_id = stations[i]['station_id']
        district = stations[i]['district']
        district_id = stations[i]['district_id']
        time_zone = stations[i]['time_zone']
        data={}
        data['utc']=[]
        data['local_t']=[]
        data['temperature']=[]
        data['humidity']=[]
        data['wind_direction']=[]
        data['wind_speed/kph']=[]
        
        
        #2018/05/04增加判断文件是否已经存在，如果存在则认为已经抓取不重复抓取
        if check_requested:
            file = '{}.csv'.format(station_id)
            file = os.path.join(folder,file)
            if os.path.isfile(file):
                continue
        #-----------------------------------------------
        
        local_hour = now_hour+time_zone+1
        if local_hour > 24:
            local_hour = local_hour%24
        
        for i in range(math.ceil(predict_hours/hours_step)):   
            hour = i*hours_step+local_hour
            url = 'https://www.accuweather.com/en/cn/{}/{}/hourly-weather-forecast/{}?hour={}'.format(
                    district,district_id,district_id,hour)
            print(url)
            request = urllib.request.Request(url,headers=headers)
            response = urllib.request.urlopen(request)
            rescode = response.getcode()
            if(rescode==200):
                soup = BeautifulSoup(response,"html.parser")
                table1 = soup.find('div',{'class':'hourly-table overview-hourly'})
                table3 = soup.find('div',{'class':'hourly-table sky-hourly'})
                
                trs = table1.findAll('tr')
                for tr in trs:
                    if tr.th.string == 'Temp (°C)':
                        temps = tr.findAll('span')
                        for temp in temps:
                            data['temperature'].append(str(temp.string)[:-1])
                            
                    elif tr.th.string == 'Wind (km/h)':
                        winds = tr.findAll('span')
                        for wind in winds:
                            strs = str(wind.string).split(' ')
                            data['wind_speed/kph'].append(strs[0])
                            data['wind_direction'].append(winddirection[strs[1]])
                            
                    
                    
                trs = table3.findAll('tr')
                for tr in trs:
                    if tr.th.string == 'SKY':
                        times = tr.findAll('div')
                        t=0
                        for _time in times:
                            data['local_t'].append(str(_time.string))
                            data['utc'].append((hour-time_zone+t)%24)
                            t += 1
                            
                    elif tr.th.string == 'Humidity':
                        humiditys = tr.findAll('span')
                        for humidity in humiditys:
                            data['humidity'].append(str(humidity.string)[:-1])                    
                
            else:
                print("Error Code:" + rescode)
                return
            
        file = '{}.csv'.format(station_id)
        file = os.path.join(folder,file)
        data_pd = pd.DataFrame.from_dict(data)
        data_pd.to_csv(file)
        #data_pd.to_csv(file,columns=['utc','local_t','temperature','humidity','wind_direction','wind_speed/kph','weather'])

def request_from_caiyun(cities, start, folder):
    start = start - datetime.timedelta(hours=1)
    start_time = start.strftime("%Y-%m-%d-%H")
    folder = os.path.join(folder, 'caiyun')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for city in cities:
        save_file = os.path.join(folder, city+'_grid'+'.csv')
        url = 'http://kdd.caiyunapp.com/competition/forecast/{}/{}/2k0d1d8'.format(city, start_time)
        print(url)
        respones= requests.get(url)
        with open (save_file,'w') as f:
            f.write(respones.text)
    





