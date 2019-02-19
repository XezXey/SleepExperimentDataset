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
import errno
import seaborn as sns; sns.set(color_codes=True)


subject_folder = sys.argv[1]
#subject_folder = "Subject06"

#if len(sys.argv) == 1:
#    sys.exit("No subject input")

print("Data from : " + subject_folder)
path = './' + subject_folder + '*/All_Device_Preprocess/*.csv'


devices_filename = glob.glob(path)

# Removing raw filename for ignore in from visualising
raw_filename = glob.glob('./' + subject_folder + '*/All_Device_Preprocess/*_raw.csv')
biosppy_filename = glob.glob('./' + subject_folder + '*/All_Device_Preprocess/*_biosppy.csv')
try:
    devices_filename.remove(raw_filename[0])
    devices_filename.remove(biosppy_filename[0])
    devices_filename.remove(biosppy_filename[1])
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
    if find_filename(filename) == 'biosignalsplux':
        devices_list_df.append(pd.read_csv(filename, index_col=None))
        devices_dict_df[find_filename(filename)] = pd.read_csv(filename, index_col=None)
    else:
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
interest_cols = ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica', 'HR_biosignalsplux']
devices_df = devices_df.loc[:, interest_cols].groupby(devices_df['Timestamp']).mean()
devices_df['Timestamp'] = devices_df.index.time

# Plotting
# Slicing into the interest interval
start_time = "13:00:10"
end_time = "16:38:23"
start_time_obj = dt.datetime.strptime(start_time, "%H:%M:%S").replace(microsecond=0).time()
end_time_obj = dt.datetime.strptime(end_time, "%H:%M:%S").replace(microsecond=0).time()
devices_df_interval = devices_df.loc[(devices_df['Timestamp'] > start_time_obj) & (devices_df['Timestamp'] < end_time_obj)]

# Slicing into the resting and sleeping state
start_time_resting = devices_df['AX_empatica'].dropna().index[0]
end_time_resting = start_time_resting + dt.timedelta(minutes=30)
start_time_sleeping = end_time_resting + dt.timedelta(minutes=5)
end_time_sleeping = devices_df['AX_empatica'].dropna().index[-1]
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

path_img = './Visualization_Image/CompareLineScatter/' + subject_folder + '/'
# Trying to make directory if it's not exist
if not os.path.exists(os.path.dirname(path_img)):
    try:
        os.makedirs(os.path.dirname(path_img))
    except OSError as exc: #Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


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
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
fig.suptitle('All devices with all states', fontsize=50)
plt.subplots_adjust(top=0.88,
bottom=0.05,
left=0.05,
right=0.98,
hspace=0.225,
wspace=0.155)
ax1_hr = plt.subplot(2, 1, 1)
ax2_acc = plt.subplot(2, 1, 2)
for each_device in devices_df_interval_plot_hr.columns:
    #print(each_device)
    #ax1_hr.set_xticklabels(devices_df_interval_plot_hr[each_device].dropna().index.time, rotation=45)
    ax1_hr.plot(devices_df_interval_plot_hr[each_device].dropna().index.time, devices_df_interval_plot_hr[each_device].dropna(), 'x-', markersize=0.8)
    
ax1_hr.set_title("Heart rate measure by wearable devices")
ax1_hr.set_ylabel("Heart rate(bpm)")
ax1_hr.set_xlabel("Time")
ax1_hr.legend()

ax2_acc.set_title("Acceleration measure by empatica")
ax2_acc.plot(devcies_df_interval_plot_acc.index.time, devcies_df_interval_plot_acc)
ax2_acc.legend(devcies_df_interval_plot_acc.columns)
ax2_acc.set_ylabel("Accelerometer")
ax2_acc.set_xlabel("Time")
plt.savefig(path_img + subject_folder + '_all_devices_states', quality=95)
#plt.show()

# 2. Line plot : ACC-HR (by state)

acc_col = ['AX_empatica', 'AY_empatica', 'AZ_empatica']
resting_devices = ['HR_empatica', 'HR_biosignalsplux', 'HR_fitbit']
sleeping_devices = ['HR_biosignalsplux', 'HR_fitbit', 'HR_empatica', 'HR_emfitqs']
activity_devices = ['HR_fitbit', 'HR_polarh10', 'HR_empatica', 'HR_applewatch', 'HR_ticwatch']

state_devices = {'resting' : resting_devices, 'sleeping' : sleeping_devices, 'activity' : activity_devices}
state_df_hr = {'resting' : devices_df_interval_resting, 'sleeping' : devices_df_interval_sleeping, 'activity' : devices_df_interval_activity}
    
graph_count_for_save = 0
for state_name, state_value_list in state_devices.items():
    my_dpi = 96
    fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
    plt.subplots_adjust(top=0.88,
        bottom=0.05,
        left=0.05,
        right=0.98,
        hspace=0.225,
        wspace=0.155)
    ax1_hr = plt.subplot(2, 1, 1)
    ax2_acc = plt.subplot(2, 1, 2)
    for each_device in state_devices[state_name]:
        #print(each_device)
        #ax1_hr.set_xticklabels(devices_df_interval_plot_hr[each_device].dropna().index.time, rotation=45)
        ax1_hr.plot(state_df_hr[state_name][each_device].dropna().index.time, state_df_hr[state_name][each_device].dropna(), 'x-', markersize=0.8)
        
    fig.suptitle(state_name + ' state', fontsize=50)
    ax1_hr.set_title("Heart Rate measure by wearable devices")
    ax1_hr.set_ylabel("Heart rate(bpm)")
    ax1_hr.set_xlabel("Time")
    ax1_hr.legend()
    
    ax2_acc.set_title("Acceleration measure by empatica")
    ax2_acc.plot(state_df_hr[state_name][acc_col].index.time, state_df_hr[state_name][acc_col])
    ax2_acc.legend(state_df_hr[state_name][acc_col])
    ax2_acc.set_ylabel("Accelerometer")
    ax2_acc.set_xlabel("Time")
    plt.savefig(path_img + subject_folder + '_' + state_name + '_' + str(graph_count_for_save) , quality=95)
    graph_count_for_save += 1
    

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
#graph_count_for_save = 0
for state_name, state_value_list in state_devices.items():
    my_dpi = 96
    fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
    plt.subplots_adjust(top=0.88,
        bottom=0.05,
        left=0.05,
        right=0.98,
        hspace=0.225,
        wspace=0.155)
    for index, device in enumerate(state_value_list):
        for acc_axis in acc_col:
            graph_count = graph_count + 1
            #print(device + ' in ' + state_name)
            axes_state = plt.subplot(len(state_value_list), len(acc_col), graph_count)
            axes_state.scatter(state_df_hr[state_name][acc_axis], state_df_hr[state_name][device], s=1.5)
            axes_state.set_xlabel(acc_axis)
            axes_state.set_ylabel(device)
    fig.suptitle(state_name + ' state', fontsize=50)
    #plt.show()
    graph_count = 0
    plt.savefig(path_img + subject_folder + '_' + state_name + '_' + str(graph_count_for_save) , quality=95)
    graph_count_for_save += 1
    

# Plotting by include all devices per acc_col
graph_count = 0
#graph_count_for_save = 0
for state_name, state_value_list in state_devices.items():
    my_dpi = 96
    fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
    plt.subplots_adjust(top=0.88,
        bottom=0.05,
        left=0.05,
        right=0.98,
        hspace=0.225,
        wspace=0.155)
    for acc_axis in acc_col:    
        graph_count = graph_count + 1
        axes_state = plt.subplot(1, len(acc_col), graph_count)
        for index, device in enumerate(state_value_list):
            #print(device + ' in ' + state_name)
            axes_state.scatter(state_df_hr[state_name][acc_axis], state_df_hr[state_name][device], s=1.5)
            axes_state.set_xlabel(acc_axis)
            axes_state.set_ylabel('Heart rate(bpm)')
    axes_state.legend(loc=0)
    fig.suptitle(state_name + ' state', fontsize=50)
    #plt.show()
    plt.savefig(path_img + subject_folder + '_' + state_name + '_' + str(graph_count_for_save) , quality=95)
    graph_count_for_save += 1
    graph_count = 0

#plt.show()


# 4. Plotting HR graph and error with freedson
path_img = './Visualization_Image/ErrorWithFreedson/' + subject_folder + '/'
# Trying to make directory if it's not exist
if not os.path.exists(os.path.dirname(path_img)):
    try:
        os.makedirs(os.path.dirname(path_img))
    except OSError as exc: #Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
devices_df_interval_all_states_freedson = pd.concat([devices_df_interval_resting, devices_df_interval_sleeping, devices_df_interval_activity])
devices_df_interval_resting_freedson = devices_df_interval_resting
devices_df_interval_sleeping_freedson = devices_df_interval_sleeping
devices_df_interval_activity_freedson = devices_df_interval_activity


#plt.plot((devices_df_interval_all_states_freedson['HR_biosignalsplux']-devices_df_interval_all_states_freedson['HR_fitbit']).dropna(), 'x-', markersize=1.5)
# 1.Empatica

# Resting states
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Empatica Error with Freedson(PA level) in Resting states', fontsize=40)
axes_resting_freedsod_empatica = plt.subplot(3, 1, 1)
empatica_resting_reg_plot = (devices_df_interval_resting_freedson['HR_biosignalsplux']-devices_df_interval_resting_freedson['HR_empatica']).where(devices_df_interval_resting_freedson['HR_biosignalsplux']-devices_df_interval_resting_freedson['HR_empatica'] < 50, np.nan)
try:
    sns.regplot(y = empatica_resting_reg_plot, x=list(range(0, len(empatica_resting_reg_plot.index))), ax=axes_resting_freedsod_empatica)
    axes_resting_freedsod_empatica.axhline(0, ls='--', color='red')
    axes_resting_freedsod_empatica.set(ylabel='Empatica - Error Heart rate(bmp)', xlabel='Records')
    axes_resting_freedsod_empatica = plt.subplot(3, 1, 2)
    axes_resting_freedsod_empatica.plot(devices_df_interval_resting_freedson.index.time, devices_df_interval_resting_freedson['PA_lvl_VectorA_empatica_encoded'])
    axes_resting_freedsod_empatica.set(ylabel='Physical Activity Level', xlabel='Time')
    axes_resting_freedsod_empatica = plt.subplot(3, 1, 3)
    axes_resting_freedsod_empatica.set(ylabel='Accelerometer', xlabel='Time')
    axes_resting_freedsod_empatica.plot(devices_df_interval_resting_freedson.index.time, (devices_df_interval_resting_freedson['VectorA_empatica']))
except ValueError:
    print('No data matching found on' + subject_folder + ' - Empatica Error with Freedson(PA level) in Resting states')

fig.savefig(path_img + subject_folder + '_empatica_resting_states')

# Sleeping States
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Empatica Error with Freedson(PA level) in Sleeping states', fontsize=40)
axes_sleeping_freedsod_empatica = plt.subplot(3, 1, 1)
empatica_sleeping_reg_plot = (devices_df_interval_sleeping_freedson['HR_biosignalsplux']-devices_df_interval_sleeping_freedson['HR_empatica']).where(devices_df_interval_sleeping_freedson['HR_biosignalsplux']-devices_df_interval_sleeping_freedson['HR_empatica'] < 50, np.nan)
axes_sleeping_freedsod_empatica.axhline(0, ls='--', color='red')
sns.regplot(y = empatica_sleeping_reg_plot, x=list(range(0, len(empatica_sleeping_reg_plot.index))), ax=axes_sleeping_freedsod_empatica)
axes_sleeping_freedsod_empatica.set(ylabel='Empatica - Error Heart rate(bmp)', xlabel='Records')
axes_sleeping_freedsod_empatica = plt.subplot(3, 1, 2)
axes_sleeping_freedsod_empatica.plot(devices_df_interval_sleeping_freedson.index.time, devices_df_interval_sleeping_freedson['PA_lvl_VectorA_empatica_encoded'])
axes_sleeping_freedsod_empatica.set(ylabel='Physical Activity Level', xlabel='Time')
axes_sleeping_freedsod_empatica = plt.subplot(3, 1, 3)
axes_sleeping_freedsod_empatica.plot(devices_df_interval_sleeping_freedson.index.time, (devices_df_interval_sleeping_freedson['VectorA_empatica']))
axes_sleeping_freedsod_empatica.set(ylabel='Accelerometer', xlabel='Time')
fig.savefig(path_img + subject_folder + '_empatica_sleeping_states')

# Activity States
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Empatica Error with Freedson(PA level) in Activity states', fontsize=40)
empatica_activity_reg_plot = (devices_df_interval_activity_freedson['HR_polarh10']-devices_df_interval_activity_freedson['HR_empatica']).where(devices_df_interval_activity_freedson['HR_polarh10']-devices_df_interval_activity_freedson['HR_empatica'] < 50, np.nan)
axes_activity_freedsod_empatica = plt.subplot(3, 1, 1)
sns.regplot(y = devices_df_interval_activity_freedson['HR_polarh10']-devices_df_interval_activity_freedson['HR_empatica'], x=list(range(0, len(devices_df_interval_activity_freedson.index))), ax=axes_activity_freedsod_empatica)
axes_activity_freedsod_empatica.axhline(0, ls='--', color='red')
axes_activity_freedsod_empatica.set(ylabel='Empatica - Error Heart rate(bmp)', xlabel='Records')
axes_activity_freedsod_empatica = plt.subplot(3, 1, 2)
axes_activity_freedsod_empatica.plot(devices_df_interval_activity_freedson.index.time, devices_df_interval_activity_freedson['PA_lvl_VectorA_empatica_encoded'])
axes_activity_freedsod_empatica.set(ylabel='Physical Activity Level', xlabel='Time')
axes_activity_freedsod_empatica = plt.subplot(3, 1, 3)
axes_activity_freedsod_empatica.plot(devices_df_interval_activity_freedson.index.time, (devices_df_interval_activity_freedson['VectorA_empatica']))
axes_activity_freedsod_empatica.set(ylabel='Accelerometer', xlabel='Time')
fig.savefig(path_img + subject_folder + '_empatica_activity_states')

# 2.Fitbit
# Resting states
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Fitbit Error with Freedson(PA level) in Resting states', fontsize=40)
axes_resting_freedsod_fitbit = plt.subplot(3, 1, 1)
fitbit_resting_reg_plot = (devices_df_interval_resting_freedson['HR_biosignalsplux']-devices_df_interval_resting_freedson['HR_fitbit']).where(devices_df_interval_resting_freedson['HR_biosignalsplux']-devices_df_interval_resting_freedson['HR_fitbit'] < 50, np.nan)
try:
    sns.regplot(y = fitbit_resting_reg_plot, x=list(range(0, len(fitbit_resting_reg_plot.index))), ax=axes_resting_freedsod_fitbit)
    axes_resting_freedsod_fitbit.axhline(0, ls='--', color='red')
    axes_resting_freedsod_fitbit.set(ylabel='Fitbit - Error Heart rate(bmp)', xlabel='Records')
    axes_resting_freedsod_fitbit = plt.subplot(3, 1, 2)
    axes_resting_freedsod_fitbit.plot(devices_df_interval_resting_freedson.index.time, devices_df_interval_resting_freedson['PA_lvl_VectorA_empatica_encoded'])
    axes_resting_freedsod_fitbit.set(ylabel='Physical Activity Level', xlabel='Time')
    axes_resting_freedsod_fitbit = plt.subplot(3, 1, 3)
    axes_resting_freedsod_fitbit.plot(devices_df_interval_resting_freedson.index.time, (devices_df_interval_resting_freedson['VectorA_empatica']))
    axes_resting_freedsod_fitbit.set(ylabel='Accelerometer', xlabel='Time')

except ValueError:
    print('No data matching found on ' + subject_folder + ' - Fitbit Error with Freedson(PA level) in Resting states')
fig.savefig(path_img + subject_folder + '_fitbit_resting_states')

# Sleeping States
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Fitbit Error with Freedson(PA level) in Sleeping states', fontsize=40)
fitbit_sleeping_reg_plot = (devices_df_interval_sleeping_freedson['HR_biosignalsplux']-devices_df_interval_sleeping_freedson['HR_fitbit']).where(devices_df_interval_sleeping_freedson['HR_biosignalsplux']-devices_df_interval_sleeping_freedson['HR_fitbit'] < 50, np.nan)
axes_sleeping_freedsod_fitbit = plt.subplot(3, 1, 1)
sns.regplot(y = fitbit_sleeping_reg_plot, x=list(range(0, len(fitbit_sleeping_reg_plot.index))), ax=axes_sleeping_freedsod_fitbit)
axes_sleeping_freedsod_fitbit.axhline(0, ls='--', color='red')
axes_sleeping_freedsod_fitbit.set(ylabel='Fitbit - Error Heart rate(bmp)', xlabel='Records')
axes_sleeping_freedsod_fitbit = plt.subplot(3, 1, 2)
axes_sleeping_freedsod_fitbit.plot(devices_df_interval_sleeping_freedson.index.time, devices_df_interval_sleeping_freedson['PA_lvl_VectorA_empatica_encoded'])
axes_sleeping_freedsod_fitbit.set(ylabel='Physical Activity Level', xlabel='Time')
axes_sleeping_freedsod_fitbit = plt.subplot(3, 1, 3)
axes_sleeping_freedsod_fitbit.plot(devices_df_interval_sleeping_freedson.index.time, (devices_df_interval_sleeping_freedson['VectorA_empatica']))
axes_sleeping_freedsod_fitbit.set(ylabel='Accelerometer', xlabel='Time')
fig.savefig(path_img + subject_folder + '_fitbit_sleeping_states')


# Activity States
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Fitbit Error with Freedson(PA level) in Activity states', fontsize=40)
fitbit_activity_reg_plot = (devices_df_interval_activity_freedson['HR_polarh10']-devices_df_interval_activity_freedson['HR_fitbit']).where(devices_df_interval_activity_freedson['HR_polarh10']-devices_df_interval_activity_freedson['HR_fitbit'] < 50, np.nan)
axes_activity_freedsod_fitbit = plt.subplot(3, 1, 1)
sns.regplot(y = fitbit_activity_reg_plot, x=list(range(0, len(fitbit_activity_reg_plot.index))), ax=axes_activity_freedsod_fitbit)
axes_activity_freedsod_fitbit.axhline(0, ls='--', color='red')
axes_activity_freedsod_fitbit.set(ylabel='Fitbit - Error Heart rate(bmp)', xlabel='Records')
axes_activity_freedsod_fitbit = plt.subplot(3, 1, 2)
axes_activity_freedsod_fitbit.plot(devices_df_interval_activity_freedson.index.time, devices_df_interval_activity_freedson['PA_lvl_VectorA_empatica_encoded'])
axes_activity_freedsod_fitbit.set(ylabel='Physical Activity Level', xlabel='Time')
axes_activity_freedsod_fitbit = plt.subplot(3, 1, 3)
axes_activity_freedsod_fitbit.plot(devices_df_interval_activity_freedson.index.time, (devices_df_interval_activity_freedson['VectorA_empatica']))
axes_activity_freedsod_fitbit.set(ylabel='Accelerometer', xlabel='Time')

fig.savefig(path_img + subject_folder + '_fitbit_activity_states')

# 3.EmfitQS
# Sleeping States
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - EmfitQS Error with Freedson(PA level) in Sleeping states', fontsize=40)
emfitqs_activity_reg_plot = (devices_df_interval_sleeping_freedson['HR_biosignalsplux']-devices_df_interval_sleeping_freedson['HR_emfitqs']).where(devices_df_interval_sleeping_freedson['HR_biosignalsplux']-devices_df_interval_sleeping_freedson['HR_emfitqs'] < 50, np.nan)
axes_sleeping_freedsod_emfitqs = plt.subplot(3, 1, 1)
sns.regplot(y = emfitqs_activity_reg_plot, x=list(range(0, len(emfitqs_activity_reg_plot.index))), ax=axes_sleeping_freedsod_emfitqs)
axes_sleeping_freedsod_emfitqs.axhline(0, ls='--', color='red')
axes_sleeping_freedsod_emfitqs.set(ylabel='EmfitQS - Error Heart rate(bmp)', xlabel='Records')
axes_sleeping_freedsod_emfitqs = plt.subplot(3, 1, 2)
axes_sleeping_freedsod_emfitqs.plot(devices_df_interval_sleeping_freedson.index.time, devices_df_interval_sleeping_freedson['PA_lvl_VectorA_empatica_encoded'])
axes_sleeping_freedsod_emfitqs.set(ylabel='Physical Activity Level', xlabel='Time')
axes_sleeping_freedsod_emfitqs = plt.subplot(3, 1, 3)
axes_sleeping_freedsod_emfitqs.plot(devices_df_interval_sleeping_freedson.index.time, (devices_df_interval_sleeping_freedson['VectorA_empatica']))
axes_sleeping_freedsod_emfitqs.set(ylabel='Accelerometer', xlabel='Time')

fig.savefig(path_img + subject_folder + '_emfitqs_sleeping_states')



plt.plot((devices_df_interval_sleeping_freedson['HR_biosignalsplux']-devices_df_interval_sleeping_freedson['HR_empatica']).dropna(), 'x-', markersize=1.5)
plt.plot((devices_df_interval_sleeping_freedson['HR_biosignalsplux']-devices_df_interval_sleeping_freedson['HR_fitbit']).dropna(), 'x-', markersize=1.5)
plt.plot((devices_df_interval_sleeping_freedson['HR_biosignalsplux']-devices_df_interval_sleeping_freedson['HR_emfitqs']).dropna(), 'x-', markersize=1.5)

plt.plot((devices_df_interval_sleeping_freedson['PA_lvl_VectorA_empatica_encoded']))
plt.plot((devices_df_interval_sleeping_freedson['VectorA_empatica']))

'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica'

plt.plot(devices_df_interval_sleeping['HR_biosignalsplux']-devices_df_interval_sleeping['HR_fitbit'], 'o-')
plt.plot(devices_df_interval_resting['HR_biosignalsplux']-devices_df_interval_resting['HR_fitbit'], 'o-')

