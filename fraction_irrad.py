# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 22:05:20 2021

@author: x51b783
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def footprint_area(FOV, altitude):
    
    radius = np.multiply(altitude,np.tan(np.radians(np.divide(FOV,2))))
    area = np.multiply(np.pi,np.square(radius))
    return radius, area

altitudes = [10, 30]

df = pd.DataFrame(columns = ['Altitude (m)', 'FOV', 'Footprint Diameter (m)','Irradiance Fraction'])

for altitude in altitudes:
    df_add = pd.DataFrame(columns = ['Altitude (m)', 'FOV', 'Footprint Diameter (m)','Irradiance Fraction'])
    
    FOVs = np.arange(0,175,.5)
    radii, areas = footprint_area(FOVs, altitude)
    fractions = np.square(np.sin(np.arctan(np.divide(radii,altitude))))
    
    df_add['Altitude (m)'] = [altitude]*len(FOVs)
    df_add['FOV'] = FOVs
    df_add['Footprint Diameter (m)'] = np.multiply(radii,2)
    df_add['Irradiance Fraction'] = fractions
    
    df = df.append(df_add, ignore_index=True)
    





sns.set(rc={'figure.figsize':(3.35,4.2)}) #fig size in inches
sns.set(font="Arial")
sns.set_style("whitegrid")
sns.set_style('ticks')
fig, ax1 = plt.subplots()
degree_sign = u"\N{DEGREE SIGN}"
pallete = sns.color_palette("colorblind", n_colors = 13)

#ax1.set_title('Average Percipitation Percentage by Month', fontsize=16)
ln1 = sns.lineplot(ax = ax1, data = df.loc[df['Altitude (m)'] == 10], x='FOV', y='Footprint Diameter (m)', color=pallete[0], 
                   label='Footprint Diameter, 10 m AGL')

sns.lineplot(ax = ax1, data = df.loc[df['Altitude (m)'] == 30], x='FOV', y='Footprint Diameter (m)', color=pallete[5], 
                   label='Footprint Diameter, 30 m AGL')

#specify we want to share the same x-axis
ax2 = ax1.twinx()

#line plot creation
#ax2.set_ylabel('Avg Percipitation %', fontsize=16)


ln2 = sns.lineplot(ax = ax2, data = df.loc[df['Altitude (m)'] == 10], x='FOV', y='Irradiance Fraction', color=pallete[7], 
                   label='Irradiance Fraction', linestyle='dashed')



ax1.set_xlabel('FOV (' + degree_sign + ')', fontsize=12)
ax1.set_ylabel('Footprint diameter (m)', fontsize=12)
ax2.set_ylabel('Fraction of Irradiance', fontsize=12)

plt.xticks(ha='center')
ax1.grid(False)
ax2.grid(False)

lines_1, labels_1 = ln1.get_legend_handles_labels()
lines_2, labels_2 = ln2.get_legend_handles_labels()

lines = lines_1 + lines_2
labels = labels_1 + labels_2

ax1.legend(lines, labels, loc=0, fontsize=8)

ax2.get_legend().remove()
#show plot
plt.show()

fig.savefig('C:/Users/x51b783/Documents/Mirror/Masters/writing/frontiers_figures/fraction_irradiance.png', bbox_inches="tight",dpi=300)

