# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:40:16 2021

@author: x51b783
"""
import pandas as pd
import pytz

csv_path = 'C:/Temp/IMU/3-11-2021 11-55-07 AM.csv'

sensor_height=1.5
df = pd.read_csv(csv_path, index_col = [0])
df.index.name = 'datetime'


for index, row in df.iterrows():
    if(df.loc[index, 'tilt_dir'] >180):
        df.loc[index, 'adj_tilt_dir'] = df.loc[index, 'tilt_dir']-180
    else:
        df.loc[index, 'adj_tilt_dir'] = df.loc[index, 'tilt_dir']+180


df['longitude'] = [-111.4758549 for x in range (df.shape[0])]
df['latitude'] = [45.23182628 for x in range (df.shape[0])]
df['elevation'] = [2633.215677 + sensor_height for x in range (df.shape[0])]

df.to_csv(csv_path)


