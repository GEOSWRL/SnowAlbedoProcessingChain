# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:26:53 2021

@author: x51b783
"""

import pandas as pd
import pytz
import os
import earthpy as ep
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def tilt_direction_to_azimuth(a):
    """
    Creates new Geotiff from numpy array

    Parameters
    ----------
    a : integer angle from -180 to 180 degrees
    
    Returns
    -------
    a : angle corrected to 360 degree azimuth (clockwise from 0 degrees at North)

    """
    
    if(a<0):
        a+=360
        return a
    return a

#statistics = 'C:/Temp/0318/YC20210318_statistics.csv'
csv_all_path = 'C:/Users/x51b783/Documents/Mirror/Masters/Class/field_measurement_snow/project/SnowExData/20210224_albedo_sd.csv'

csv_path = 'C:/Temp/IMU/3-11-2021 11-48-35 AM.csv'
csv2_path = 'C:/Users/x51b783/Documents/Mirror/Masters/Class/field_measurement_snow/project/SnowExData/20210224_albedo_wtf.csv'
csv3_path = 'C:/Users/x51b783/Documents/Mirror/Masters/Class/field_measurement_snow/project/SnowExData/FieldC_0217.csv'

csv_bg = 'C:/Temp/0318/merged/albedo_10m_bg.csv'
csv_USGS = 'C:/Users/x51b783/Documents/Mirror/Masters/Class/field_measurement_snow/project/SnowExData/20210217_albedo_wtf.txt'
#wd_path = os.path.join(ep.io.HOME, 'Documents', 'Mirror', 'SnowAlbedoProcessingChain', 'working_directory_test', 'logfiles', 'merged','50m_merged.csv')
if not (os.path.exists(csv_path)):
    print('working directory does not exist')
    
df = pd.read_csv(csv_path)
#df2 = pd.read_csv(csv2_path, index_col=0)
#df3 = pd.read_csv(csv3_path, index_col=0)

#df = df.append(df2)
#df = df.append(df3)

#df_bg = pd.read_csv(csv_bg)
#df_USGS = pd.read_csv(csv_USGS)
#df_stats = pd.read_csv(statistics)
"""
df_filtered = df.loc[df['vel'] <= 2.1]
df_filtered = df_filtered.loc[df['cos_avg_ls'] >= 0]
df_filtered = df_filtered.loc[df['vel'] >= 1.5]
df_filtered = df_filtered.loc[df['incoming (W/m^2)'] >=200]
"""

#print(df.dtypes)
#df['IMU_ATTI(0):tiltDirectionEarthFrame:c']
"""
g = sns.JointGrid(data=df, x=df['mean_aspect'], y=df['cos_avg_aspect'], space=0)
g.plot_joint(sns.kdeplot,
             fill=True, clip=((110, 165), (110, 165)),
             thresh=0, levels=100, cmap="rocket")
g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)
"""


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)

"""
sns.violinplot(data=[df['albedo'], df2["albedo"]], 
                     palette="Set3", bw=.2, cut=1, linewidth=1, inner="quartile")
#ax.set_ylim([0.28,1.1])
ax.set_xticklabels(['Feb 17','Feb 24'])
"""

#sns.boxplot(data=[df["cosine_avg_correctedo_albedo"], df['cos_avg_ls8']], palette="Set3", linewidth=1)

####################### scatter plot ########################################
#sns.scatterplot(data = df, x='alt_agl', y='rmse_ls8', hue = 'elev_source', s=200, palette = 'hls', style = 'Date')
#sns.scatterplot(y=df['albedo'], x=df.index, s=200, palette = 'hls')

sns.scatterplot(y=df['albedo'], x=df.index, s=200, palette = 'hls')
sns.scatterplot(y=df['corrected_albedo_ss'], x=df.index, s=200, palette = 'hls')

#sns.scatterplot(y=df['cos_avg_aspect'], x=df.index, s=200, palette = 'hls')
#sns.scatterplot(y=df['tilt_dir'], x=df.index, s=200, palette = 'hls')

#sns.scatterplot(y=df['incoming (W/m^2)'], x=df.index, s=200, palette = 'hls')
#sns.scatterplot(y=df['reflected (W/m^2)'], x=df.index, s=200, palette = 'hls')

#sns.scatterplot(y=df['cos_avg_slope'], x=df.index, s=200, palette = 'hls')
#sns.scatterplot(y=df['tilt'], x=df.index, s=200, palette = 'hls')

#sns.scatterplot(y=np.degrees(np.arccos(df['cos_sensor_incidence_angle'])), x=df.index, s=200, palette = 'hls')
#sns.scatterplot(y=np.degrees(np.arccos(df['cos_slope_incidence_angle'])), x=df.index, s=200, palette = 'hls')


#sns.kdeplot(x=df['cos_avg_ls8'], y=df['cosine_avg_correctedo_albedo'], levels=5, color="w", linewidths=1)

#sns.scatterplot(y=df['incoming (W/m^2)'], x=np.degrees(np.arccos(df['cos_sensor_incidence_angle'])), s=200, palette = 'hls')



"""
h = sns.jointplot(data=df_filtered, x='cos_avg_ls', y='cosine_avg', kind="hex", color="#4CB391")
h.ax_joint.set_xlabel('Average Snow Depth (m)', fontsize='x-large')
h.ax_joint.set_ylabel('Albedo', fontsize='x-large')
"""

"""
x=df_filtered['cos_avg_ls'].tolist()
x=np.array(x).reshape((-1,1))
y=df_filtered['cos_avg']
model = LinearRegression().fit(x, y)

print('intercept:', model.intercept_)
print('slope:', model.coef_)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
"""
#sns.scatterplot(x=df['alt_agl'], y=df['bias_ls8'], hue = df['elev_source'], s=200, palette = 'hls', legend = None)
#sns.scatterplot(x=df['alt_agl'], y=df['sd_corrected'], hue = df['elev_source'])
#sns.scatterplot(x=df['alt_agl'], y=df['corrected_mean'], hue = df['elev_source'])

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#ax.xaxis.set_label_text(40)
#ax.set_xlabel('Snow Depth (m)', fontsize = 'xx-large')
ax.set_ylabel('Albedo', fontsize = 'xx-large')
#ax.legend(title = 'legend', fontsize = 'x-large', title_fontsize = 'xx-large')
#ax.legend(None)



