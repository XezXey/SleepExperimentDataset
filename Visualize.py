#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:23:21 2019
Preprocess the device data file
@author: puntawat
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

subject_folder = sys.argv[1]
#subject_folder = "Subject10"

if len(sys.argv) == 1:
    sys.exit("No subject input")

print("Data from : " + subject_folder)
path = './' + subject_folder + '*/All_Device_Preprocess/*.csv'


devices_filename = glob.glob(path)

# Removing raw filename for ignore in from visualising
raw_filename = glob.glob('./' + subject_folder + '*/All_Device_Preprocess/*_raw.csv')
try:
    devices_filename.remove(raw_filename[0])
except IndexError or ValueError:
    print('Everything is fine. Nothing to be remove')
    
    
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
    elif 'biosignalsplux' in filename:
        return 'biosignalsplux'


devices_dict_df = {}
devices_list_df = []
for index, filename in enumerate(devices_filename):
    #print(index)
    #print(filename)
    devices_list_df.append(pd.read_csv(filename, index_col=0))
    devices_dict_df[find_filename(filename)] = pd.read_csv(filename, index_col=0)
"""
for each_device in devices_dict_df.keys():￼
    print(devices_dict_df[each_device].head(3))
    print(devices_dict_df[each_device].info())
"""
devices_df = pd.concat(devices_list_df, ignore_index=True, sort=True) # sort = True : For retaining the current behavior and silence the warning, pass 'sort=True'.

# Take millisecond part out and parse to datetime object
devices_df['Timestamp'] = devices_df['Timestamp'].apply(lambda each_time : dt.datetime.strptime(each_time.split('.')[0], '%Y-%m-%d %H:%M:%S').replace(microsecond=0))
devices_df = devices_df.sort_values(by=['Timestamp'], ascending=True).reset_index(drop = True)
cols = devices_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
devices_df = devices_df[cols]
devices_df = devices_df.loc[:, ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'HR_biosignalsplux']].groupby(devices_df['Timestamp']).mean()
devices_df['Timestamp'] = devices_df.index.time

# Plotting
# Slicing into the interest interval
start_time = "13:00:10"
end_time = "16:38:23"
start_time_obj = dt.datetime.strptime(start_time, "%H:%M:%S").replace(microsecond=0).time()
end_time_obj = dt.datetime.strptime(end_time, "%H:%M:%S").replace(microsecond=0).time()
devices_df_interval = devices_df.loc[(devices_df['Timestamp'] > start_time_obj) & (devices_df['Timestamp'] < end_time_obj)]

# Slicing into the resting and sleeping state
start_time_resting = devices_df['HR_biosignalsplux'].dropna().index[0]
end_time_resting = start_time_resting + dt.timedelta(minutes=30)
start_time_sleeping = end_time_resting + dt.timedelta(minutes=5)
end_time_sleeping = devices_df['HR_biosignalsplux'].dropna().index[-1]
start_time_activity = devices_df['HR_polarh10'].dropna().index[0]
end_time_activity = devices_df['HR_polarh10'].dropna().index[-1]

# For analyze
devices_df_interval_resting = devices_df.loc[(devices_df['Timestamp'] > start_time_resting.time()) & (devices_df['Timestamp'] < end_time_resting.time())]
devices_df_interval_sleeping = devices_df.loc[(devices_df['Timestamp'] > start_time_sleeping.time()) & (devices_df['Timestamp'] < end_time_sleeping.time())]
devices_df_interval_activity = devices_df.loc[(devices_df['Timestamp'] > start_time_activity.time()) & (devices_df['Timestamp'] < end_time_activity.time())]

"""
# Writing to csv file for only grouped 

subject_folder = glob.glob(subject_folder + '*')[0]
#if subject_folder == []:
#    sys.exit("Cannot find that subject")

#subject_folder = 'Subject01_2019-1-16'

path_grouped = './' + subject_folder + '/All_Device_Grouped/'
# Trying to make directory if it's not exist
if not os.path.exists(os.path.dirname(path_grouped)):
    try:
        os.makedirs(os.path.dirname(path_grouped))
    except OSError as exc: #Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
devices_df_interval_resting.to_csv(path_grouped + subject_folder + '_grouped_resting.csv')
devices_df_interval_sleeping.to_csv(path_grouped + subject_folder + '_grouped_sleeping.csv')
devices_df_interval_activity.to_csv(path_grouped + subject_folder + '_grouped_activity.csv')
"""

# 1. Line plot : HR and ACC
#lineObjects = plt.plot(devices_df_interval['Timestamp'], devices_df_interval[devices_df_interval.iloc[:, :-1].columns], marker='x', markersize=0.8, linestyle='-')

# Select Columns for plot
# 1. For all columns
devices_df_interval_plot = devices_df_interval[devices_df_interval.iloc[:, :-1].columns]
# 2. For specific columns
devices_df_interval_plot_hr = devices_df_interval_plot[['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 'HR_biosignalsplux']]

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

ax2_acc.set_title("Acceleration measure by empatica")
ax2_acc.plot(devcies_df_interval_plot_acc.index.time, devcies_df_interval_plot_acc)
ax2_acc.legend(devcies_df_interval_plot_acc.columns)
ax2_acc.set_ylabel("Accelerometer")
ax2_acc.set_xlabel("Time")
plt.show()

# 2. Line plot : ACC-HR (by state)

acc_col = ['AX_empatica', 'AY_empatica', 'AZ_empatica']
resting_devices = ['HR_empatica', 'HR_biosignalsplux', 'HR_fitbit']
sleeping_devices = ['HR_biosignalsplux', 'HR_fitbit', 'HR_empatica', 'HR_emfitqs']
activity_devices = ['HR_fitbit', 'HR_polarh10', 'HR_empatica', 'HR_applewatch', 'HR_ticwatch']

state_devices = {'resting' : resting_devices, 'sleeping' : sleeping_devices, 'activity' : activity_devices}
state_df_hr = {'resting' : devices_df_interval_resting, 'sleeping' : devices_df_interval_sleeping, 'activity' : devices_df_interval_activity}
    
for state_name, state_value_list in state_devices.items():
    fig = plt.figure()

    ax1_hr = plt.subplot(2, 1, 1)
    ax2_acc = plt.subplot(2, 1, 2)
    for each_device in state_devices[state_name]:
        #print(each_device)
        #ax1_hr.set_xticklabels(devices_df_interval_plot_hr[each_device].dropna().index.time, rotation=45)
        ax1_hr.plot(state_df_hr[state_name][each_device].dropna().index.time, state_df_hr[state_name][each_device].dropna(), 'x-', markersize=0.8)
        
    fig.suptitle(state_name + ' state', fontsize=50)
    ax1_hr.set_title("HeartRate measure by wearable devices")
    ax1_hr.set_ylabel("Heartrate(bpm)")
    ax1_hr.set_xlabel("Time")
    ax1_hr.legend()
    
    ax2_acc.set_title("Acceleration measure by empatica")
    ax2_acc.plot(state_df_hr[state_name][acc_col].index.time, state_df_hr[state_name][acc_col])
    ax2_acc.legend(state_df_hr[state_name][acc_col])
    ax2_acc.set_ylabel("Accelerometer")
    ax2_acc.set_xlabel("Time")
    fig.savefig(state_name + ' state')
    

#    plt.show()


# 3. Scattter plot : HR and ACC


acc_col = ['AX_empatica', 'AY_empatica', 'AZ_empatica']
resting_devices = ['HR_empatica', 'HR_biosignalsplux', 'HR_fitbit']
sleeping_devices = ['HR_biosignalsplux', 'HR_fitbit', 'HR_empatica', 'HR_emfitqs']
activity_devices = ['HR_fitbit', 'HR_polarh10', 'HR_empatica', 'HR_applewatch', 'HR_ticwatch']

state_devices = {'resting' : resting_devices, 'sleeping' : sleeping_devices, 'activity' : activity_devices}
state_df_hr = {'resting' : devices_df_interval_resting, 'sleeping' : devices_df_interval_sleeping, 'activity' : devices_df_interval_activity}

fig = plt.figure()

# Plotting by seperate devices
graph_count = 0
for state_name, state_value_list in state_devices.items():
    fig = plt.figure()
    for index, device in enumerate(state_value_list):
        for acc_axis in acc_col:
            graph_count = graph_count + 1
            print(device + ' in ' + state_name)
            axes_state = plt.subplot(len(state_value_list), len(acc_col), graph_count)
            axes_state.scatter(state_df_hr[state_name][acc_axis], state_df_hr[state_name][device], s=1.5)
            axes_state.set_xlabel(acc_axis)
            axes_state.set_ylabel(device)
    fig.suptitle(state_name + ' state', fontsize=50)
    #plt.show()
    graph_count = 0
    fig.savefig(state_name + '_state')
    

# Plotting by include all devices per acc_col
graph_count = 0
for state_name, state_value_list in state_devices.items():
    fig = plt.figure()

    for acc_axis in acc_col:    
        graph_count = graph_count + 1
        axes_state = plt.subplot(1, len(acc_col), graph_count)
        for index, device in enumerate(state_value_list):
            print(device + ' in ' + state_name)
            axes_state.scatter(state_df_hr[state_name][acc_axis], state_df_hr[state_name][device], s=1.5)
            axes_state.set_xlabel(acc_axis)
            axes_state.set_ylabel('Heartrate(bpm)')
    axes_state.legend(loc=0)
    fig.suptitle(state_name + ' state', fontsize=50)
    #plt.show()
    fig.savefig(state_name + '_state')

    graph_count = 0

plt.show()

"""
plt.scatter(devices_df_interval_resting['AX_empatica'], devices_df_interval_resting['HR_biosignalsplux'], s=1.5)
plt.scatter(devices_df_interval_resting['AY_empatica'], devices_df_interval_resting['HR_biosignalsplux'], s=1.5)
plt.scatter(devices_df_interval_resting['AZ_empatica'], devices_df_interval_resting['HR_biosignalsplux'], s=1.5)
plt.legend()


plt.scatter(devices_df_interval_sleeping['AX_empatica'], devices_df_interval_sleeping['HR_biosignalsplux'], s=1.5)
plt.scatter(devices_df_interval_sleeping['AY_empatica'], devices_df_interval_sleeping['HR_biosignalsplux'], s=1.5)
plt.scatter(devices_df_interval_sleeping['AZ_empatica'], devices_df_interval_sleeping['HR_biosignalsplux'], s=1.5)

plt.scatter(devices_df_interval_activity['AX_empatica'], devices_df_interval_activity['HR_polarh10'], s=1.5)
plt.scatter(devices_df_interval_activity['AY_empatica'], devices_df_interval_activity['HR_polarh10'], s=1.5)
plt.scatter(devices_df_interval_activity['AZ_empatica'], devices_df_interval_activity['HR_polarh10'], s=1.5)
"""