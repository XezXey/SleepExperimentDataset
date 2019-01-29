#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:23:21 2019
Preprocess the device data file
@author: puntawat
"""
from blaze import Data, compute, by
import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

subject_folder = sys.argv[1]
#subject_folder = "Subject01"
"""
if len(sys.argv) == 1:
    sys.exit("No subject input")
"""
print("Data from : " + subject_folder)
path = './' + subject_folder + '*/All_Device_Preprocess/*.csv'


devices_filename = glob.glob(path)
print(devices_filename)
if len(devices_filename) == 6:
    devices_list = ['applewatch', 'fitbit', 'emfitqs', 'empatica', 'polarh10', 'ticwatch']
else : devices_list = ['applewatch', 'fitbit', 'emfitqs', 'empatica', 'polarh10']

devices_dict_df = {}
devices_list_df = []
for index, filename in enumerate(devices_filename):
    print(index)
    devices_list_df.append(pd.read_csv(filename, index_col=0))
    devices_dict_df[devices_list[index]] = pd.read_csv(filename, index_col=0)
    
for each_device in devices_list:
    print(devices_dict_df[each_device].head(3))
    print(devices_dict_df[each_device].info())

devices_df = pd.concat(devices_list_df, ignore_index=True)
# Take millisecond part out and parse to datetime object
devices_df['Timestamp'] = devices_df['Timestamp'].apply(lambda each_time : dt.datetime.strptime(each_time.split('.')[0], '%Y-%m-%d %H:%M:%S').replace(microsecond=0))
devices_df = devices_df.sort_values(by=['Timestamp'], ascending=True).reset_index(drop = True)
cols = devices_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
devices_df = devices_df[cols]
#devices_df = devices_df.loc[:, ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_fitbit', 'HR_emfitqs']].groupby(devices_df['Timestamp'].apply(lambda each_time : dt.datetime.strptime(each_time.split('.')[0], '%Y-%m-%d %H:%M:%S').replace(microsecond=0)).transform('mean'))
devices_df = devices_df.loc[:, ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_Ticwatch']].groupby(devices_df['Timestamp']).mean()
devices_df['Timestamp'] = devices_df.index.time


# Plotting
start_time = dt.datetime.strptime("15:35:23", "%H:%M:%S").replace(microsecond=0).time()
end_time = dt.datetime.strptime("17:35:23", "%H:%M:%S").replace(microsecond=0).time()
devices_df_interval = devices_df.loc[(devices_df['Timestamp'] > start_time) & (devices_df['Timestamp'] < end_time)]
plt.plot(devices_df_interval['Timestamp'], devices_df_interval[['HR_applewatch', 'HR_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_Ticwatch', 'HR_polarh10']])
plt.show()
# Groupby Timestamp
# pd.groupby(devices_dict_df['applewatch'],)

"""
# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(devices_df.index.time, devices_df['HR_emfitqs'], 'r-') # Returns a tuple of line objects, thus the comma

for phase in np.linspace(0, 10*np.pi, 500):
    line1.set_ydata(np.sin(x + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()
    
""" 