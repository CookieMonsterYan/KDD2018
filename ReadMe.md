
KDD2018

The solution of our Team 迟到大队.

data_pre.py: 训练数据地预处理
models.py: 定义了网络结构
train_models.py：训练模型
run_predict.py：进行预报并自动提交结果

We used LSTM as our basic solution. We trained the basic model with all data in a city, then trained models for each station with the basic model and data of the station.

We also tried using part of the train set as well as other structures.

In the practice stage, we used the weather data from accuweather and openweather. In the final stage we used the official caiyun weather api.

As we have several models for each station, we tried to combine these models based on their scores of last days. However, in most days, these stratagems showed no better scores.

