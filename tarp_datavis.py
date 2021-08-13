# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:38:50 2021

@author: x51b783
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

path_to_tarp = 'D:/field_data/tarp/tarps_plot.csv'

df = pd.read_csv(path_to_tarp)
pallete = sns.color_palette('colorblind', 4)
pallete_hex = pallete.as_hex()
sns.set(rc={'figure.figsize':(6,4)}) #fig size in inches
sns.set(font="Arial")
sns.set_theme(style="ticks")
scatterplot = sns.scatterplot(data = df, x = 'Height (m)', y = 'Albedo', hue = 'Surface', palette = pallete)
scatterplot.set_xlabel('Height (m)', fontsize = 12)
scatterplot.set_ylabel('Albedo', fontsize = 12)

plt.savefig('D:/field_data/tarp/tarps_plot.tiff', bbox_inches="tight",dpi=300)