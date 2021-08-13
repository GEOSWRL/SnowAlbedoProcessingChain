# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:12:10 2021

@author: x51b783
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

data_dirs=['D:/field_data/YC/YC20210311/merged/', 
          'D:/field_data/YC/YC20210318/merged/', 
          'D:/field_data/YC/YC20210428/merged/',
          ]

######################################### scatter/lineplot #################################################
df_summary=pd.DataFrame(columns = ['Uncorrected RMSE', 'RMSE Corrected Albedo', 'AGL Altitude (m)', 'Topography Source', 'Date'])
'''
for data_dir in data_dirs: 
    for filename in os.listdir(data_dir):
        
        
        topo_source = ''
        rmse_uncorrected=0
        rmse_corrected=0
        
        if filename.endswith('.csv'):
            
            df = pd.read_csv(data_dir+filename)
            rmse_uncorrected = ((df['albedo'] - df['cos_avg_ls8_ss']) ** 2).mean() ** .5
            rmse_corrected = ((df['corrected_albedo_ss'] - df['cos_avg_ls8_ss']) ** 2).mean() ** .5
            
            if filename.endswith('m.csv'):
                topo_source = 'Snow Surface SfM'
            
            if filename.endswith('bg.csv'):
                topo_source = 'Bareground SfM'
            
            if filename.endswith('USGS.csv'):
                topo_source = '3DEP 1/3 Arc Second'
                
            df_summary = df_summary.append({'Uncorrected RMSE': rmse_uncorrected,
                                            'RMSE Corrected Albedo': rmse_corrected,
                                            'AGL Altitude (m)':int(filename[7:9]),
                                            'Topography Source':topo_source, 
                                            'Date': data_dir[19:27]}, ignore_index = True)

sns.set(rc={'figure.figsize':(3.35,2.8)}) #fig size in inches
#sns.set(rc={'figure.figsize':(7,4)})
sns.set(font="Arial")
sns.set_theme(style="ticks")  
sns.set_palette(sns.color_palette('colorblind', 10)) 
            

        
ax = sns.scatterplot(data = df_summary, x = 'AGL Altitude (m)', y = 'RMSE Corrected Albedo', hue = 'Topography Source', style= 'Date',s=80)
ax = sns.lineplot(data = df_summary, x = 'AGL Altitude (m)', y = 'RMSE Corrected Albedo', hue = 'Topography Source', style= 'Date', alpha = 0.5, legend = False)
#ax = sns.scatterplot(data = df_summary, x = 'AGL Altitude', y = 'Uncorrected RMSE', hue = 'Topo Source', style= 'Date')
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set(ylim = (0, 0.25))
ax.set_ylabel('RMSE Corrected Albedo', fontsize=12)
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)
#plt.savefig('C:/Users/x51b783/Documents/Mirror/Masters/writing/frontiers_figures/topo_source_altitude2.tiff', bbox_inches="tight",dpi=300)
###############################################################################################################
'''
###################################################### topography source comparison #######################################
data_dir = 'D:/field_data/YC/YC20210318/merged/'

df_summary=pd.DataFrame(columns = ['Uncorrected Albedo', 'Corrected Albedo', 
                                   'LS8 Albedo', 'AGL Altitude', 'Topography Source', 'Date'])

 
for filename in os.listdir(data_dir):
        
        
    topo_source = ''
    rmse_uncorrected=0
    rmse_corrected=0
        
    if filename.endswith('.csv'):
            
        df = pd.read_csv(data_dir+filename)
            
        if filename.endswith('m.csv'):
            topo_source = 'Snow Surface SfM'
            
        if filename.endswith('bg.csv'):
            topo_source = 'Bareground SfM'
            
        if filename.endswith('USGS.csv'):
            topo_source = '3DEP 1/3 Arc Second'
                
        df_summary = df_summary.append(pd.DataFrame({'Uncorrected Albedo': df['albedo'],
                                            'Corrected Albedo': df['corrected_albedo_ss'],
                                            'LS8 Albedo': df['cos_avg_ls8_ss'],
                                            'AGL Altitude':[int(filename[7:9])]* len(df),
                                            'Topography Source':[topo_source] * len(df), 
                                            'Date': [data_dir[19:27]]*len(df)}), ignore_index = True)


df_filtered = df_summary.loc[df_summary['AGL Altitude'] == 20]

df_ss = df_filtered.loc[df_filtered['Topography Source'] == 'Snow Surface SfM']['Corrected Albedo']
df_bg = df_filtered.loc[df_filtered['Topography Source'] == 'Bareground SfM']['Corrected Albedo'].to_numpy()
df_usgs = df_filtered.loc[df_filtered['Topography Source'] == '3DEP 1/3 Arc Second']['Corrected Albedo'].to_numpy()

df_ls8 = df_filtered.loc[df_summary['Topography Source'] == 'Snow Surface SfM']['LS8 Albedo']
df_uncorr = df_filtered.loc[df_summary['Topography Source'] == 'Snow Surface SfM']['Uncorrected Albedo']

df_plot = pd.DataFrame(columns = ['Uncorrected Albedo', 'Snow Surface SfM', 'Bareground SfM', '3DEP 1/3 Arc Second', 'LS8 Albedo'])

df_plot['Uncorrected Albedo'] = df_uncorr
df_plot['Snow Surface SfM'] = df_ss
df_plot['Bareground SfM'] = df_bg
df_plot['3DEP 1/3 Arc Second'] = df_usgs
df_plot['LS8 Albedo'] = df_ls8
            

ax = sns.boxplot(data = df_plot)
ax.set_title('Mar. 18, 2021 - 20 m AGL', fontsize=12)
ax.set_ylabel('Albedo')
plt.setp(ax.get_xticklabels(), rotation=90)

plt.savefig('C:/Users/x51b783/Documents/Mirror/Masters/writing/frontiers_figures/topo_sourceB.tiff', bbox_inches="tight",dpi=300)

##########################################################











