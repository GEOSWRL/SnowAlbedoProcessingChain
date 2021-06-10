# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:06:34 2021

@author: x51b783
"""
import pandas as pd
import numpy as np



def calc_statistics(csv_path, agl_alt):
#calculate mean, standard deviation, RMSE and bias

    df = pd.read_csv(csv_path)

    mean_uncorrected = np.mean(df['albedo'])
    mean_corrected = np.mean(df['cosine_avg_correctedo_albedo'])
    mean_ls8 = np.mean(df['cos_avg_ls8'])

    sd_uncorrected = np.std(df['albedo'])
    sd_corrected = np.std(df['cosine_avg_correctedo_albedo'])
    sd_ls8 = np.std(df['cos_avg_ls8'])

    rmse_ls8 = np.sqrt(np.divide(np.sum(np.square(np.subtract(df['cosine_avg_correctedo_albedo'],df['cos_avg_ls8']))), df.shape[0]))

    bias_ls8 = np.mean(np.subtract(df['cos_avg_ls8'], df['cosine_avg_correctedo_albedo']))
    
    return {'alt_agl': agl_alt, 'uncorrected_mean': mean_uncorrected, 'corrected_mean': mean_corrected, 'mean_ls8': mean_ls8,
            'sd_uncorrected': sd_uncorrected, 'sd_corrected': sd_corrected, 'sd_ls8': sd_ls8,
            'rmse_ls8': rmse_ls8, 'bias_ls8': bias_ls8}

df = pd.DataFrame()
df = df.append(calc_statistics('C:/Temp/0311/merged/albedo_10m_USGS.csv',10), ignore_index = True)
df = df.append(calc_statistics('C:/Temp/0311/merged/albedo_15m_USGS.csv',15), ignore_index = True)
df = df.append(calc_statistics('C:/Temp/0311/merged/albedo_20m_USGS.csv',20), ignore_index = True)

df.to_csv('C:/Temp/0311/YC20210311_USGS_statistics.csv')