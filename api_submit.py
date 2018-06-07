
# coding: utf-8

import requests


def submit(file, description="submit"):
    files={'files': open(file,'rb')}
    
    data = {
        "user_id": "frozencookie",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
        "team_token": "35aa5090c41b258a14d8261e654d6abd660206417a91eba10eb4fa9cdae831d6", #your team_token.
        "description": description,  #no more than 40 chars.
        "filename": file, #your filename
    }
    
    url = 'https://biendata.com/competition/kdd_2018_submit/'
    
    response = requests.post(url, files=files, data=data)
    
    print(response.text)


