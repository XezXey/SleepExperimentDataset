#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:39:50 2019

@author: puntawat
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import glob
import os
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns; sns.set(color_codes=True)
import errno

subject_folder = sys.argv[1]
#subject_folder = "Subject07"

if len(sys.argv) == 1:
    sys.exit("No subject input")

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
    print('--->Everything is fine. Nothing to be remove')
    
    
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

devices_df = pd.concat(devices_list_df, ignore_index=True, sort=True) # sort = True : For retaining the current behavior and silence the warning, pass 'sort=True'.
list_hr_features = ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 'HR_biosignalsplux']
# Filter the heart rate over 200 out
print("--->Filter the outlier data...")
for each_hr_feature in list_hr_features:
    try :
        devices_df.loc[devices_df[each_hr_feature] > 200, each_hr_feature] = np.nan
        devices_df.loc[devices_df[each_hr_feature] < 40, each_hr_feature] = np.nan
    except KeyError:
        print('------>No ' + each_hr_feature + ' features found')
# Take millisecond part out and parse to datetime object
devices_df['Timestamp'] = devices_df['Timestamp'].apply(lambda each_time : dt.datetime.strptime(each_time.split('.')[0], '%Y-%m-%d %H:%M:%S').replace(microsecond=0))
devices_df = devices_df.sort_values(by=['Timestamp'], ascending=True).reset_index(drop = True)
cols = devices_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
devices_df = devices_df[cols]
interest_cols = ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica', 'HR_biosignalsplux']
devices_df = devices_df.loc[:, interest_cols].groupby(devices_df['Timestamp']).mean()
devices_df['Timestamp'] = devices_df.index.time


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
devices_df_interval_activity = devices_df.loc[(devices_df['Timestamp'] > start_time_activity.time()) & (devices_df['Timestamp'] < end_time_activity.time())]

devices_df.to_csv(path_grouped + subject_folder + '_grouped_all_states.csv')
devices_df_interval_resting.to_csv(path_grouped + subject_folder + '_grouped_resting.csv')
devices_df_interval_sleeping.to_csv(path_grouped + subject_folder + '_grouped_sleeping.csv')
devices_df_interval_activity.to_csv(path_grouped + subject_folder + '_grouped_activity.csv')
