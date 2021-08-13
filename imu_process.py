# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:38:42 2021

@author: x51b783
"""

import numpy as np
import pandas as pd
import topo_correction_util as tcu
import angles

yaw_offset = 56

#clean data and compress

#df = pd.read_csv('C:/Temp/IMU/YC20210311_IMU_data.txt', delimiter = '\s+', skiprows=[0])
df = pd.read_csv('D:/field_data/YC/YC20210318/imu/YC20210318_imu_validation.txt', delimiter = ',', skiprows=[0])



df.index = df["Record Time:"]

df = df.drop(columns = ['ChipTime:', 'ax：', 'ay：', 'az：', 'wx：', 'wy：', 'wz：', 'Unnamed: 11'])

df['AngleY：'] = df['AngleY：']

df = df.groupby(df.index).mean()

#calculate tilt and tilt direction
tilt = [angles.get_tilt_witmotion(pitch, roll, yaw)[0] for (pitch, roll, yaw) in zip(df['AngleY：'], df['AngleX：'], df['AngleZ：']-yaw_offset)]

tilt_dir = [angles.get_tilt_witmotion(pitch, roll, yaw)[1] for (pitch, roll, yaw) in zip(df['AngleY：'], df['AngleX：'], df['AngleZ：']-yaw_offset)]

df['tilt'] = tilt
df['tilt_dir'] = tilt_dir

df.index = pd.DatetimeIndex(df.index)
#df.to_csv('C:/Temp/IMU/Record2_processed.txt')


df_meteon = pd.read_excel('D:/field_data/YC/YC20210318/imu/YC20210318_validation.xls', usecols=[0,2,5], skiprows=9, 
                          names = ["Time", "incoming (W/m^2)", "reflected (W/m^2)"], parse_dates=True, index_col = 'Time',
                          sheet_name = None)

for key in df_meteon:
    
    incoming = df_meteon[key]['incoming (W/m^2)']
    reflected = df_meteon[key]['reflected (W/m^2)']
    df_meteon[key]['albedo'] = reflected / incoming
    df_meteon[key] = pd.concat([df_meteon[key], df], axis=1, join='inner')
    df_meteon[key]
    df_meteon[key].to_csv('D:/field_data/YC/YC20210318/imu/' + key +'.csv')
