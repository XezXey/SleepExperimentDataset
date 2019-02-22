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
import errno

subject_folder = sys.argv[1]
#subject_folder = "Subject09"

#if len(sys.argv) == 1:
#    sys.exit("No subject input")

print("Visualizing One By One, Data from : " + subject_folder)
path = './' + subject_folder + '*/All_Device_Grouped/*.csv'

devices_filename = glob.glob(path)
for each_fn in devices_filename:
    if 'grouped_all_states' in each_fn:
        grouped_all_devices_fn = each_fn
        #print(each_fn)
        break
devices_df = pd.read_csv(grouped_all_devices_fn)
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
start_time_resting = devices_df['AX_empatica'].dropna().index[0]
end_time_resting = start_time_resting + dt.timedelta(minutes=30)
start_time_sleeping = end_time_resting + dt.timedelta(minutes=5)
#end_time_sleeping = devices_df['HR_biosignalsplux'].dropna().index[-1]
end_time_sleeping = start_time_sleeping + dt.timedelta(minutes=90)
start_time_activity = devices_df['HR_polarh10'].dropna().index[0]
end_time_activity = devices_df['HR_polarh10'].dropna().index[-1]

# For analyze
devices_df_interval_resting = devices_df.loc[(devices_df['Timestamp'] > start_time_resting.time()) & (devices_df['Timestamp'] < end_time_resting.time())]
devices_df_interval_sleeping = devices_df.loc[(devices_df['Timestamp'] > start_time_sleeping.time()) & (devices_df['Timestamp'] < end_time_sleeping.time())]
real_end_of_sleeping_index = devices_df_interval_sleeping['AX_empatica'].dropna().index[-1]
devices_df_interval_sleeping = devices_df_interval_sleeping.loc[devices_df_interval_sleeping.index < real_end_of_sleeping_index]
devices_df_interval_activity = devices_df.loc[(devices_df['Timestamp'] > start_time_activity.time()) & (devices_df['Timestamp'] < end_time_activity.time())]


# Create folder to store image
path_img = './Visualization_Image/OneByOne/' + subject_folder + '/'
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
axes_resting.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_fitbit(bpm)')
try:
    sns.regplot(x=devices_df_interval_resting['HR_biosignalsplux'], y=devices_df_interval_resting['HR_fitbit'], ax=axes_resting)
    max_limit = max(devices_df_interval_resting[['HR_biosignalsplux', 'HR_fitbit']].dropna().max())+1
    min_limit = min(devices_df_interval_resting[['HR_biosignalsplux', 'HR_fitbit']].dropna().min())-1
    axes_resting.set(xlim=(min_limit, None), ylim=(min_limit, None))
except ValueError:
    print("--->(Resting)Fitbit records are not matching with biosignalsplux")


axes_resting = plt.subplot(2, 2, 2)
axes_resting.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_empatica(bpm)')
try:
    sns.regplot(x=devices_df_interval_resting['HR_biosignalsplux'], y=devices_df_interval_resting['HR_empatica'], ax=axes_resting)
    max_limit = max(devices_df_interval_resting[['HR_biosignalsplux', 'HR_empatica']].dropna().max())+1
    min_limit = min(devices_df_interval_resting[['HR_biosignalsplux', 'HR_empatica']].dropna().min())-1
    axes_resting.set(xlim=(min_limit, None), ylim=(min_limit, None))
except ValueError:
    print("--->(Resting)Empatica records are not matching with biosignalsplux")

axes_resting = plt.subplot(2, 2, 3)
try :
    sns.regplot(x=devices_df_interval_resting['HR_biosignalsplux'], y=devices_df_interval_resting['HR_fitbit'], label='HR_fitbit', ax=axes_resting)
    max_limit = [max(devices_df_interval_resting[['HR_biosignalsplux', 'HR_empatica']].dropna().max())+1, 
                 max(devices_df_interval_resting[['HR_biosignalsplux', 'HR_fitbit']].dropna().max())+1]
    max_limit = max(max_limit)
    min_limit = [min(devices_df_interval_resting[['HR_biosignalsplux', 'HR_empatica']].dropna().min())-1, 
                 min(devices_df_interval_resting[['HR_biosignalsplux', 'HR_fitbit']].dropna().min())-1]
    min_limit = min(min_limit)
    axes_resting.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
    print('--->(Resting)Fitbit records are not matching with biosignalsplux')
try:
    sns.regplot(x=devices_df_interval_resting['HR_biosignalsplux'], y=devices_df_interval_resting['HR_empatica'], ax=axes_resting)
except ValueError:
    print("--->(Resting)Empatica records are not matching with biosignalsplux")
axes_resting.legend()


fig.savefig(path_img + subject_folder + '_resting_state', quality=95)


# Sleeping state
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
axes_sleeping.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_fitbit(bpm)')
try:
    sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_fitbit'], ax=axes_sleeping)
    max_limit = max(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_fitbit']].dropna().max())+1
    min_limit = min(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_fitbit']].dropna().min())-1
    axes_sleeping.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
    print('--->(Sleeping)Fitbit records are not matching with biosignalsplux')

axes_sleeping = plt.subplot(2, 2, 2)
axes_sleeping.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_empatica(bpm)')
try:
    sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_empatica'], ax=axes_sleeping)
    max_limit = max(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_empatica']].dropna().max())+1
    min_limit = min(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_empatica']].dropna().min())-1
    axes_sleeping.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
    print('--->(Sleeping)Emapatica records are not matching with biosignalsplux')


axes_sleeping = plt.subplot(2, 2, 3)
axes_sleeping.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_emfitqs(bpm)')
try:
    sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_emfitqs'], ax=axes_sleeping)
    max_limit = max(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_emfitqs']].dropna().max())+1
    min_limit = min(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_emfitqs']].dropna().min())-1
    axes_sleeping.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
    print('--->(Sleeping)Emfitqs records are not matching with biosignalpslux')

axes_sleeping = plt.subplot(2, 2, 4)
axes_sleeping.set(xlabel='ECG_HR_biosignalsplux(bpm)', ylabel='HR_all_devices(bpm)')
try:
    sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_fitbit'], label='HR_fitbit', ax=axes_sleeping)
    max_limit = [max(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_empatica']].dropna().max())+1, 
                 max(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_fitbit']].dropna().max())+1, 
                 max(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_emfitqs']].dropna().max())+1]
    max_limit = max(max_limit)
    min_limit = [min(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_empatica']].dropna().min())-1, 
                 min(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_fitbit']].dropna().min())-1, 
                 min(devices_df_interval_sleeping[['HR_biosignalsplux', 'HR_emfitqs']].dropna().min())+1]
    min_limit = min(min_limit)
    axes_resting.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
    print('--->(Sleeping)Fitbit records are not matching with biosignalpslux')
try:
    sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_empatica'], label='HR_empatica', ax=axes_sleeping)
except ValueError:
    print('--->(Sleeping)Empatica records are not matching with biosignalpslux')
    
try:
    sns.regplot(x=devices_df_interval_sleeping['HR_biosignalsplux'], y=devices_df_interval_sleeping['HR_emfitqs'], label='HR_emfitqs', ax=axes_sleeping)

except ValueError:
    print('--->(Sleeping)EmfitQS records are not matching with biosignalpslux')
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
axes_activity.set(xlabel='HR_polarh10(bpm)', ylabel='HR_fitbit(bpm)')
try:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_fitbit'], ax=axes_activity)
    max_limit = max(devices_df_interval_activity[['HR_polarh10', 'HR_fitbit']].dropna().max())+1
    min_limit = min(devices_df_interval_activity[['HR_polarh10', 'HR_fitbit']].dropna().min())-1
    axes_activity.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
    print('--->(Activity)Fitbit records are not matching with polarh10')


axes_activity = plt.subplot(3, 2, 2)
axes_activity.set(xlabel='HR_polarh10(bpm)', ylabel='HR_empatica(bpm)')
try:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_empatica'], ax=axes_activity)
    max_limit = max(devices_df_interval_activity[['HR_polarh10', 'HR_empatica']].dropna().max())+1
    min_limit = min(devices_df_interval_activity[['HR_polarh10', 'HR_empatica']].dropna().min())-1
    axes_activity.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
        print('--->(Activity)Empatica records are not matching with polarh10')



axes_activity = plt.subplot(3, 2, 3)
axes_activity.set(xlabel='HR_polarh10(bpm)', ylabel='HR_applewatch(bpm)')
try:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_applewatch'], ax=axes_activity)
    max_limit = max(devices_df_interval_activity[['HR_polarh10', 'HR_applewatch']].dropna().max())+1
    min_limit = min(devices_df_interval_activity[['HR_polarh10', 'HR_applewatch']].dropna().min())-1
    axes_activity.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
    print('--->(Activity)Applewatch records are not matching with polarh10')


axes_activity = plt.subplot(3, 2, 4)
axes_activity.set(xlabel='HR_polarh10(bpm)', ylabel='HR_ticwatch(bpm)')
try:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_ticwatch'], label='HR_ticwatch', ax=axes_activity)
    max_limit = max(devices_df_interval_activity[['HR_polarh10', 'HR_ticwatch']].dropna().max())+1
    min_limit = min(devices_df_interval_activity[['HR_polarh10', 'HR_ticwatch']].dropna().min())-1
    axes_activity.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
    print("--->(Activity)Ticwatch records not matching with polarh10")


axes_activity = plt.subplot(3, 2, 5)
axes_activity.set(xlabel='HR_polarh10(bpm)', ylabel='All devices(bpm)')
try:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_empatica'], label='HR_empatica', ax=axes_activity)
    max_limit = [max(devices_df_interval_activity[['HR_polarh10', 'HR_fitbit']].dropna().max())+1, 
             max(devices_df_interval_activity[['HR_polarh10', 'HR_ticwatch']].dropna().max())+1, 
             max(devices_df_interval_activity[['HR_polarh10', 'HR_empatica']].dropna().max())+1, 
             max(devices_df_interval_activity[['HR_polarh10', 'HR_applewatch']].dropna().max())+1,]
    max_limit = max(max_limit)
    min_limit = [min(devices_df_interval_activity[['HR_polarh10', 'HR_fitbit']].dropna().min())-1, 
             min(devices_df_interval_activity[['HR_polarh10', 'HR_ticwatch']].dropna().min())-1, 
             min(devices_df_interval_activity[['HR_polarh10', 'HR_empatica']].dropna().min())-1, 
             min(devices_df_interval_activity[['HR_polarh10', 'HR_applewatch']].dropna().min())-1]
    min_limit = min(min_limit)
    axes_activity.set(xlim=(min_limit, max_limit), ylim=(min_limit, max_limit))
except ValueError:
    print('--->(Activity)Empatica records are not matching with polarh10')
try:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_applewatch'], label='HR_applewatch', ax=axes_activity)
except ValueError:
    print('--->(Activity)Applewatch records are not matching with polarh10')
try:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_fitbit'], label='HR_fitbit', ax=axes_activity)
except ValueError:
    print('--->(Activity)Fitbit records are not matching with polarh10')
try:
    sns.regplot(x=devices_df_interval_activity['HR_polarh10'], y=devices_df_interval_activity['HR_ticwatch'], label='HR_ticwatch', ax=axes_activity)
except ValueError :
    print("--->(Activity)Ticwatch records not matching with polarh10")
plt.legend()

fig.savefig(path_img + subject_folder + '_activity_state', quality=95)

"""
r = devices_df_interval_resting.corr(method='pearson')

s = devices_df_interval_sleeping.corr(method='pearson')

a = devices_df_interval_activity.corr(method='pearson')
"""