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
import seaborn as sns; sns.set(color_codes=True)


subject_folder = sys.argv[1]
#subject_folder = "Subject02"

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


# Create folder to store image
path_img = './Visualization_Image/OneByOne/'
# Trying to make directory if it's not exist
if not os.path.exists(os.path.dirname(path_img)):
    try:
        os.makedirs(os.path.dirname(path_img))
    except OSError as exc: #Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Resting state
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Resting state', fontsize=50)
axes_resting = plt.subplot(2, 2, 1)
sns.regplot(x=devices_df_interval_resting['HR_biosignalsplux'], y=devices_df_interval_resting['HR_fitbit'], ax=axes_resting)
axes_resting.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_fitbit(bpm)')



axes_resting = plt.subplot(2, 2, 2)
if len(devices_df_interval_resting[['HR_empatica', 'HR_biosignalsplux']].dropna()) != 0:
    sns.regplot(x=devices_df_interval_resting['HR_biosignalsplux'], y=devices_df_interval_resting['HR_empatica'], ax=axes_resting)
else:
    print("Empatica records not matching with biosignalsplux")
axes_resting.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_empatica(bpm)')



axes_resting = plt.subplot(2, 2, 3)
sns.regplot(x=devices_df_interval_resting['HR_biosignalsplux'], y=devices_df_interval_resting['HR_fitbit'], label='HR_fitbit', ax=axes_resting)
if len(devices_df_interval_resting[['HR_empatica', 'HR_biosignalsplux']].dropna()) != 0:
    sns.regplot(x=devices_df_interval_resting['HR_biosignalsplux'], y=devices_df_interval_resting['HR_empatica'], ax=axes_resting)
else:
    print("Empatica records not matching with biosignalsplux")
axes_resting.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_all_devices(bpm)')

axes_resting.legend()

fig.savefig(path_img + subject_folder + '_resting_state', quality=95)

# Sleeping state
devices_df_interval_sleeping.loc[devices_df_interval_sleeping['HR_emfitqs'] == 0] = np.nan
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Sleeping state', fontsize=50)
axes_sleeping = plt.subplot(2, 2, 1)
sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_fitbit'], ax=axes_sleeping)
axes_sleeping.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_fitbit(bpm)')


axes_sleeping = plt.subplot(2, 2, 2)
sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_empatica'], ax=axes_sleeping)
axes_sleeping.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_empatica(bpm)')


axes_sleeping = plt.subplot(2, 2, 3)
sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_emfitqs'], ax=axes_sleeping)
axes_sleeping.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_emfitqs(bpm)')

axes_sleeping = plt.subplot(2, 2, 4)
sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_fitbit'], label='HR_fitbit', ax=axes_sleeping)
sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_empatica'], label='HR_empatica', ax=axes_sleeping)
sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_emfitqs'], label='HR_emfitqs', ax=axes_sleeping)
axes_sleeping.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_all_devices(bpm)')
plt.legend()

fig.savefig(path_img + subject_folder + '_sleeping_state', quality=95)


# Activity state
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)

fig.suptitle(subject_folder + ' - Activity state', fontsize=50)
axes_activity = plt.subplot(3, 2, 1)
sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_fitbit'], ax=axes_activity)
axes_activity.set(xlabel='HR_polarh10(bpm)', ylabel='HR_fitbit(bpm)')


axes_activity = plt.subplot(3, 2, 2)
sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_empatica'], ax=axes_activity)
axes_activity.set(xlabel='HR_polarh10(bpm)', ylabel='HR_empatica(bpm)')


axes_activity = plt.subplot(3, 2, 3)
sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_applewatch'], ax=axes_activity)
axes_activity.set(xlabel='HR_polarh10(bpm)', ylabel='HR_applewatch(bpm)')


axes_activity = plt.subplot(3, 2, 4)
if len(devices_df_interval_activity[['HR_ticwatch', 'HR_polarh10']].dropna()) != 0:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_ticwatch'], label='HR_ticwatch', ax=axes_activity)
else :
    print("Ticwatch records not matching with polarh10")
axes_activity.set(xlabel='HR_polarh10(bpm)', ylabel='HR_ticwatch(bpm)')

axes_activity = plt.subplot(3, 2, 5)
sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_empatica'], label='HR_empatica', ax=axes_activity)
sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_applewatch'], label='HR_applewatch', ax=axes_activity)
sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_fitbit'], label='HR_fitbit', ax=axes_activity)
if len(devices_df_interval_activity[['HR_ticwatch', 'HR_polarh10']].dropna()) != 0:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_ticwatch'], label='HR_ticwatch', ax=axes_activity)
else :
    print("Ticwatch records not matching with polarh10")
plt.legend()
#plt.show()
fig.savefig(path_img + subject_folder + '_activity_state', quality=95)

"""
r = devices_df_interval_resting.corr(method='pearson')

s = devices_df_interval_sleeping.corr(method='pearson')

a = devices_df_interval_activity.corr(method='pearson')
"""