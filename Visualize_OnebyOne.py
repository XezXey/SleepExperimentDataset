#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:08:02 2019

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

#subject_folder = sys.argv[1]
subject_folder = "Subject07"

#if len(sys.argv) == 1:
#    sys.exit("No subject input")

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




# Plotting OneByOne compare to the baseline(Biosignalsplux for resting&slepping and polarh10 for activity) 

resting_devices = ['HR_empatica', 'HR_fitbit']
sleeping_devices = ['HR_fitbit', 'HR_empatica', 'HR_emfitqs']
activity_devices = ['HR_fitbit', 'HR_empatica', 'HR_applewatch', 'HR_ticwatch']

state_devices = {'resting' : resting_devices, 'sleeping' : sleeping_devices, 'activity' : activity_devices}
state_df_hr = {'resting' : devices_df_interval_resting, 'sleeping' : devices_df_interval_sleeping, 'activity' : devices_df_interval_activity}


# Resting state
fig = plt.figure()
fig.suptitle('Resting state', fontsize=50)
axes_resting = plt.subplot(2, 2, 1)
axes_resting.scatter(devices_df_interval_resting['HR_biosignalsplux'], devices_df_interval_resting['HR_fitbit'], s = 5.5)
axes_resting.plot(devices_df_interval_resting['HR_biosignalsplux'],devices_df_interval_resting['HR_biosignalsplux'], 'y--')
axes_resting.set_xlabel("ECG_biosignalsplux(bpm)")
axes_resting.set_ylabel("HR_fitbit(bpm)")
axes_resting.set_xticks(np.arange(int(devices_df_interval_resting['HR_biosignalsplux'].min()), int(devices_df_interval_resting['HR_biosignalsplux'].max()), 20))
axes_resting.set_yticks(np.arange(40, 160, 20))


axes_resting = plt.subplot(2, 2, 2)
axes_resting.scatter(devices_df_interval_resting['HR_biosignalsplux'], devices_df_interval_resting['HR_empatica'], s = 5.5)
axes_resting.plot(devices_df_interval_resting['HR_biosignalsplux'],devices_df_interval_resting['HR_biosignalsplux'], 'y--')
axes_resting.set_xlabel("ECG_biosignalsplux(bpm)")
axes_resting.set_ylabel("HR_empatica(bpm)")
axes_resting.set_xticks(np.arange(int(devices_df_interval_resting['HR_biosignalsplux'].min()), int(devices_df_interval_resting['HR_biosignalsplux'].max()), 20))
axes_resting.set_yticks(np.arange(40, 160, 20))


axes_resting = plt.subplot(2, 2, 3)
axes_resting.scatter(devices_df_interval_resting['HR_biosignalsplux'], devices_df_interval_resting['HR_fitbit'], s = 5.5)
axes_resting.scatter(devices_df_interval_resting['HR_biosignalsplux'], devices_df_interval_resting['HR_empatica'], s = 5.5)
axes_resting.plot(devices_df_interval_resting['HR_biosignalsplux'],devices_df_interval_resting['HR_biosignalsplux'], 'y--')

axes_resting.set_xlabel("ECG_biosignalsplux(bpm)")
axes_resting.set_ylabel("HR_all_devices(bpm)")
axes_resting.set_xticks(np.arange(int(devices_df_interval_resting['HR_biosignalsplux'].min()), int(devices_df_interval_resting['HR_biosignalsplux'].max()), 20))
axes_resting.set_yticks(np.arange(40, 160, 20))
plt.legend()

# Sleeping state
devices_df_interval_sleeping.loc[devices_df_interval_sleeping['HR_emfitqs'] == 0] = np.nan
fig = plt.figure()
fig.suptitle('Sleeping state', fontsize=50)
axes_sleeping = plt.subplot(2, 2, 1)
axes_sleeping.scatter(devices_df_interval_sleeping['HR_biosignalsplux'], devices_df_interval_sleeping['HR_emfitqs'], s = 5.5)
axes_sleeping.plot(devices_df_interval_resting['HR_biosignalsplux'],devices_df_interval_resting['HR_biosignalsplux'], 'y--')
axes_sleeping.set_xlabel("ECG_biosignalsplux(bpm)")
axes_sleeping.set_ylabel("HR_emfitqs(bpm)")
axes_sleeping.set_xticks(np.arange(int(devices_df_interval_sleeping['HR_biosignalsplux'].min())-10, int(devices_df_interval_sleeping['HR_biosignalsplux'].max()), 20))
axes_sleeping.set_yticks(np.arange(40, 160, 20))


axes_sleeping = plt.subplot(2, 2, 2)
axes_sleeping.scatter(devices_df_interval_sleeping['HR_biosignalsplux'], devices_df_interval_sleeping['HR_empatica'], s = 5.5)
axes_sleeping.plot(devices_df_interval_resting['HR_biosignalsplux'],devices_df_interval_resting['HR_biosignalsplux'], 'y--')
axes_sleeping.set_xlabel("ECG_biosignalsplux(bpm)")
axes_sleeping.set_ylabel("HR_empatica(bpm)")
axes_sleeping.set_xticks(np.arange(int(devices_df_interval_sleeping['HR_biosignalsplux'].min())-10, int(devices_df_interval_sleeping['HR_biosignalsplux'].max()), 20))
axes_sleeping.set_yticks(np.arange(40, 160, 20))


axes_sleeping = plt.subplot(2, 2, 3)
axes_sleeping.scatter(devices_df_interval_sleeping['HR_biosignalsplux'], devices_df_interval_sleeping['HR_fitbit'], s = 5.5)
axes_sleeping.plot(devices_df_interval_resting['HR_biosignalsplux'],devices_df_interval_resting['HR_biosignalsplux'], 'y--')
axes_sleeping.set_xlabel("ECG_biosignalsplux(bpm)")
axes_sleeping.set_ylabel("HR_fitbit(bpm)")
axes_sleeping.set_xticks(np.arange(int(devices_df_interval_sleeping['HR_biosignalsplux'].min())-10, int(devices_df_interval_sleeping['HR_biosignalsplux'].max()), 20))
axes_sleeping.set_yticks(np.arange(40, 160, 20))

axes_sleeping = plt.subplot(2, 2, 4)
axes_sleeping.scatter(devices_df_interval_sleeping['HR_biosignalsplux'], devices_df_interval_sleeping['HR_emfitqs'], s = 5.5)
axes_sleeping.scatter(devices_df_interval_sleeping['HR_biosignalsplux'], devices_df_interval_sleeping['HR_empatica'], s = 5.5)
axes_sleeping.scatter(devices_df_interval_sleeping['HR_biosignalsplux'], devices_df_interval_sleeping['HR_fitbit'], s = 5.5)
axes_sleeping.plot(devices_df_interval_resting['HR_biosignalsplux'],devices_df_interval_resting['HR_biosignalsplux'], 'y--')
axes_sleeping.set_xlabel("ECG_biosignalsplux(bpm)")
axes_sleeping.set_ylabel("HR_all_devices(bpm)")
axes_sleeping.set_xticks(np.arange(int(devices_df_interval_sleeping['HR_biosignalsplux'].min())-10, int(devices_df_interval_sleeping['HR_biosignalsplux'].max()), 20))
axes_sleeping.set_yticks(np.arange(40, 160, 20))

plt.legend()


# Activity state
fig = plt.figure()
fig.suptitle('Activity state', fontsize=50)
axes_activity = plt.subplot(3, 2, 1)
axes_activity.scatter(devices_df_interval_activity['HR_polarh10'], devices_df_interval_activity['HR_applewatch'], s = 5.5)
axes_activity.plot(devices_df_interval_activity['HR_polarh10'],devices_df_interval_activity['HR_polarh10'], 'y--')

axes_activity.set_xlabel("HR_polarh10(bpm)")
axes_activity.set_ylabel("HR_applewatch(bpm)")
axes_activity.set_xticks(np.arange(int(devices_df_interval_activity['HR_polarh10'].min())-10, int(devices_df_interval_activity['HR_polarh10'].max()), 20))
axes_activity.set_yticks(np.arange(40, 160, 20))


axes_activity = plt.subplot(3, 2, 2)
axes_activity.scatter(devices_df_interval_activity['HR_polarh10'], devices_df_interval_activity['HR_ticwatch'], s = 5.5)
axes_activity.plot(devices_df_interval_activity['HR_polarh10'],devices_df_interval_activity['HR_polarh10'], 'y--')
axes_activity.set_xlabel("HR_polarh10(bpm)")
axes_activity.set_ylabel("HR_ticwatch(bpm)")
axes_activity.set_xticks(np.arange(int(devices_df_interval_activity['HR_polarh10'].min())-10, int(devices_df_interval_activity['HR_polarh10'].max()), 20))
axes_activity.set_yticks(np.arange(40, 160, 20))


axes_activity = plt.subplot(3, 2, 3)
axes_activity.scatter(devices_df_interval_activity['HR_polarh10'], devices_df_interval_activity['HR_fitbit'], s = 5.5)
axes_activity.plot(devices_df_interval_activity['HR_polarh10'],devices_df_interval_activity['HR_polarh10'], 'y--')
axes_activity.set_xlabel("HR_polarh10(bpm)")
axes_activity.set_ylabel("HR_fitbit(bpm)")
axes_activity.set_xticks(np.arange(int(devices_df_interval_activity['HR_polarh10'].min())-10, int(devices_df_interval_activity['HR_polarh10'].max()), 20))
axes_activity.set_yticks(np.arange(40, 160, 20))


axes_activity = plt.subplot(3, 2, 4)
axes_activity.scatter(devices_df_interval_activity['HR_polarh10'], devices_df_interval_activity['HR_empatica'], s = 5.5)
axes_activity.plot(devices_df_interval_activity['HR_polarh10'],devices_df_interval_activity['HR_polarh10'], 'y--')
axes_sleeping.set_xlabel("HR_polarh10(bpm)")
axes_activity.set_ylabel("HR_empatica(bpm)")
axes_activity.set_xticks(np.arange(int(devices_df_interval_activity['HR_polarh10'].min())-10, int(devices_df_interval_activity['HR_polarh10'].max()), 20))
axes_activity.set_yticks(np.arange(40, 160, 20))


axes_activity = plt.subplot(3, 2, 5)
axes_activity.scatter(devices_df_interval_activity['HR_polarh10'], devices_df_interval_activity['HR_applewatch'], s = 5.5)
axes_activity.scatter(devices_df_interval_activity['HR_polarh10'], devices_df_interval_activity['HR_ticwatch'], s = 5.5)
axes_activity.scatter(devices_df_interval_activity['HR_polarh10'], devices_df_interval_activity['HR_fitbit'], s = 5.5)
axes_activity.scatter(devices_df_interval_activity['HR_polarh10'], devices_df_interval_activity['HR_empatica'], s = 5.5)
axes_activity.plot(devices_df_interval_activity['HR_polarh10'],devices_df_interval_activity['HR_polarh10'], 'y--')
axes_activity.set_xticks(np.arange(int(devices_df_interval_activity['HR_polarh10'].min())-10, int(devices_df_interval_activity['HR_polarh10'].max()), 20))
axes_activity.set_xlabel("HR_polarh10(bpm)")
axes_activity.set_ylabel("HR_all_devices(bpm)")
axes_activity.set_yticks(np.arange(40, 160, 20))

plt.show()
plt.legend()

