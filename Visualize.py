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

#subject_folder = sys.argv[1]
subject_folder = "Subject06"

#if len(sys.argv) == 1:
#    sys.exit("No subject input")

print("Data from : " + subject_folder)
path = './' + subject_folder + '*/All_Device_Preprocess/*.csv'


devices_filename = glob.glob(path)
#print(devices_filename)
"""
if len(devices_filename) == 6:
    devices_list = ['applewatch', 'fitbit', 'emfitqs', 'empatica', 'polarh10', 'ticwatch']
else : devices_list = ['applewatch', 'fitbit', 'emfitqs', 'empatica', 'polarh10']
"""

def find_filename(filename):
    if 'applewatch' in filename:
        return 'applewatch'
    elif 'fitbit' in filename:
        return 'fitbit'
    elif 'emfitqs' in filename:
        return 'emfitqs'
    elif 'ticwatch' in filename:
        return 'ticwatch'
    elif 'polarh10' in filename:
        return 'polarh10'
    elif 'empatica' in filename:
        return 'empatica'


devices_dict_df = {}
devices_list_df = []
for index, filename in enumerate(devices_filename):
    #print(index)
    #print(filename)
    devices_list_df.append(pd.read_csv(filename, index_col=0))
    devices_dict_df[find_filename(filename)] = pd.read_csv(filename, index_col=0)
"""
for each_device in devices_dict_df.keys():
    print(devices_dict_df[each_device].head(3))
    print(devices_dict_df[each_device].info())
"""
plt.plot(devices_dict_df['empatica']['BVP_empatica'][5000:7000])
plt.show()
devices_df = pd.concat(devices_list_df, ignore_index=True, sort=True) # sort = True : For retaining the current behavior and silence the warning, pass 'sort=True'.

# Take millisecond part out and parse to datetime object
devices_df['Timestamp'] = devices_df['Timestamp'].apply(lambda each_time : dt.datetime.strptime(each_time.split('.')[0], '%Y-%m-%d %H:%M:%S').replace(microsecond=0))
devices_df = devices_df.sort_values(by=['Timestamp'], ascending=True).reset_index(drop = True)
cols = devices_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
devices_df = devices_df[cols]
devices_df = devices_df.loc[:, ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 'AX_empatica', 'AY_empatica', 'AZ_empatica']].groupby(devices_df['Timestamp']).mean()
devices_df['Timestamp'] = devices_df.index.time

# Plotting
# Slicing into the interest interval
start_time = "15:52:10"
end_time = "16:38:23"
start_time_obj = dt.datetime.strptime(start_time, "%H:%M:%S").replace(microsecond=0).time()
end_time_obj = dt.datetime.strptime(end_time, "%H:%M:%S").replace(microsecond=0).time()
devices_df_interval = devices_df.loc[(devices_df['Timestamp'] > start_time_obj) & (devices_df['Timestamp'] < end_time_obj)]
#lineObjects = plt.plot(devices_df_interval['Timestamp'], devices_df_interval[devices_df_interval.iloc[:, :-1].columns], marker='x', markersize=0.8, linestyle='-')

# Select Columns for plot
# 1. For all columns
devices_df_interval_plot = devices_df_interval[devices_df_interval.iloc[:, :-1].columns]
# 2. For specific columns
devices_df_interval_plot_hr = devices_df_interval_plot[['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch']]


# For acc plot
devcies_df_interval_plot_acc = devices_df_interval_plot[['AX_empatica', 'AY_empatica', 'AZ_empatica']]

# First try with lineObject to plot but not work cause Nan in each devices is not the same and it effect to timestamp column differently
#lineObjects = plt.plot(devices_df_interval['Timestamp'], devices_df_interval_plot_hr[devices_df_interval_plot_hr.columns], marker='x', markersize=0.8, linestyle='-')
#plt.legend(iter(lineObjects), devices_df_interval_plot_hr.columns)



# Second try to iterate plotting and it's work!!!
ax1_hr = plt.subplot(2, 1, 1)
ax2_acc = plt.subplot(2, 1, 2)

for each_device in devices_df_interval_plot_hr.columns:
    #print(each_device)
    #ax1_hr.set_xticklabels(devices_df_interval_plot_hr[each_device].dropna().index.time, rotation=45)
    ax1_hr.plot(devices_df_interval_plot_hr[each_device].dropna().index.time, devices_df_interval_plot_hr[each_device].dropna(), 'x-', markersize=0.8)
    
ax1_hr.set_title("HeartRate measure by wearable devices")
ax1_hr.set_ylabel("Heartrate(bpm)")
ax1_hr.set_xlabel("Time")
ax1_hr.legend()


"""
**For analytic purposes you can interpret the raw data as follows:
xg = x * 2/128
a value of x = 64 is in practice 1g
"""
ax2_acc.set_title("Acceleration measure by empatica")
ax2_acc.plot(devcies_df_interval_plot_acc.index.time, devcies_df_interval_plot_acc)
ax2_acc.legend(devcies_df_interval_plot_acc.columns)
ax2_acc.set_ylabel("Accelerometer")
ax2_acc.set_xlabel("Time")
plt.show()


