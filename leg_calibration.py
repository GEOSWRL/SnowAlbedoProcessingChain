# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 23:26:55 2021

@author: x51b783
"""
import pandas as pd
import pytz
import os  
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


timezone = 'US/Mountain'

path_to_Meteon = 'C:/Users/x51b783/Documents/Mirror/Masters/calibration/leg_calibration_4-8-2020.csv'

df = pd.read_csv(path_to_Meteon, usecols=[0,2,5], skiprows=9, names = ["Time", "No Leg Interference", "Leg Interference"], parse_dates=True)
#df = pd.read_csv(path_to_Meteon, usecols=[0,5,2], skiprows=9, names = ["Time", "reflected (W/m^2)", "incoming (W/m^2)"], parse_dates=True)
    
#correct downward faceing sensor for leg interference


    
#df = df.loc[(df['albedo']<=1)] #albedo cannot be > 1
    
df['Time'] = pd.DatetimeIndex(df['Time']).tz_localize(pytz.timezone(timezone))
df.set_index('Time', inplace=True)

x=np.vstack(df['Leg Interference'].to_numpy())
y=df['No Leg Interference'].to_numpy()

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(x, y)
slope = regr.coef_[0]
r2 = regr.score(x,y)
y2=1100*slope

fig, ax = plt.subplots(figsize=(3.35,2.8))

sns.set(font="Arial")
sns.set_theme(style="ticks")  
sns.set_palette(sns.color_palette("viridis", n_colors = 13)) 
ax.set(xlabel='Irradiance (W/m' + '$^2$)' + '\nwith Leg Interference', ylabel='Irradiance (W/m' + '$^2$)' + '\nwithout Leg Interference')
sns.scatterplot(data=df, x='Leg Interference', y='No Leg Interference',marker="$\circ$",ec="face", alpha = 0.7)
sns.lineplot(x=np.array([0,1100]), y=np.array([0,1122]), color='grey')
ax.text(700,.09, "$r^2$: " + str(np.round(r2,3)) + '\nslope: ' + str(np.round(slope,4)),fontsize = 10)

plt.savefig('C:/Users/x51b783/Documents/Mirror/Masters/writing/frontiers_figures/leg_calibration.tiff', bbox_inches="tight",dpi=300)