#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:23:21 2019
Preprocess the device data file
@author: puntawat
"""
# Import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

#if len(sys.argv) == 1:
#    sys.exit("No subject input")
subject_folder = sys.argv[1]
#subject_folder = "Subject07"
subject_folder = glob.glob(subject_folder + '*')[0]
if subject_folder == []:
    sys.exit("Cannot find that subject")

#subject_folder = 'Subject01_2019-1-16'

path = './' + subject_folder + '/' + 'All_Device_Preprocess/'
# Trying to make directory if it's not exist
if not os.path.exists(os.path.dirname(path)):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc: #Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Empatica Preprocess : Merge + Concat
empatica_list_df = []
empatica_list_filename = []
empatica_dict_df = {}
empatica_filename = ['Acc', 'Batt', 'Bvp', 'Gsr', 'Hr', 'Ibi', 'Tag', 'Tmp']
for fn in empatica_filename:
    empatica_list_filename = glob.glob('./' + subject_folder + '*/Empatica/subject*' + fn + '*')
    for each_fn in range(len(empatica_list_filename)):
        empatica_list_df.append(pd.read_csv(empatica_list_filename[each_fn]))
    if len(empatica_list_df) == 1:
        empatica_dict_df[fn] = empatica_list_df[0]
    else:
        empatica_dict_df[fn] = pd.concat(empatica_list_df, ignore_index = True)
    empatica_dict_df[fn] = empatica_dict_df[fn].sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)
    empatica_dict_df[fn].pop('TS_Empt')
    #plt.figure()
    #plt.title(fn)
    plt.plot(empatica_dict_df[fn].index, empatica_dict_df[fn]['TS_Machine'])
    empatica_list_df = []
    
# Merge empatica df with BVP_empatica for using the highest sampling rate    
empatica_merged_df = empatica_dict_df['Bvp']
empatica_merge_column = empatica_filename.remove('Bvp') # Remove this column name out to prevent the duplicate columns
for feature in empatica_filename:
    empatica_merged_df = empatica_merged_df.merge(empatica_dict_df[feature], on='TS_Machine', how='outer').sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)

empatica_merged_df = empatica_merged_df[pd.notnull(empatica_merged_df['TS_Machine'])]
empatica_merged_df['TS_Machine'] = empatica_merged_df['TS_Machine'].apply(lambda each_time : dt.datetime.fromtimestamp(each_time))
empatica_merged_df.rename(columns={'TS_Machine':'Timestamp', 'bvp':'BVP_empatica', 
                                   'ax':'AX_empatica', 'ay':'AY_empatica', 'az':'AZ_empatica', 
                                   'ibi':'IBI_empatica', 'gsr':'GSR_empatica', 'hr':'HR_empatica', 'tag':'TAG_empatica', 
                                   'tmp':'TEMP_empatica', 'batt':'BATT_empatica'}, inplace=True)
empatica_merged_df['HR_IBI_empatica'] = 60/empatica_merged_df['IBI_empatica']
empatica_merged_df['AX_empatica'] = empatica_merged_df['AX_empatica'] * 1/128
empatica_merged_df['AY_empatica'] = empatica_merged_df['AY_empatica'] * 1/128
empatica_merged_df['AZ_empatica'] = empatica_merged_df['AZ_empatica'] * 1/128

#bvp.bvp(empatica_merged_df['BVP_empatica'].dropna(), sampling_rate=64, show=True)
empatica_merged_df.to_csv(path + subject_folder + '_empatica.csv')


#Apply or Map 4fun
#empatica_merged_df.apply(pd.merge(x, on='TS_Machine', how='outer').sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True), empatica_list_df)
#y = y.merge(empatica_dict_df['Tmp'], on='TS_Machine', how='outer').sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)


# EmfitQS
emfitqs_list_df = []
emfitqs_list_filename = glob.glob('./' + subject_folder + '*/EmfitQS/subject*')
# Can use map here
for fn in emfitqs_list_filename:
    emfitqs_list_df.append(pd.read_csv(fn))
if len(emfitqs_list_df) > 1:
    emfitqs_concat = pd.concat(emfitqs_list_df,ignore_index=True)

emfitqs_concat.columns = list(map(lambda each_col : each_col + '_emfitqs', emfitqs_concat.columns))
emfitqs_concat = emfitqs_concat.rename(columns={'timestamp_from_machine_emfitqs':'Timestamp', })
emfitqs_concat = emfitqs_concat.sort_values(by=('Timestamp'), ascending=True).reset_index(drop=True)
emfitqs_concat['Timestamp'] = emfitqs_concat['Timestamp'].apply(lambda each_time : dt.datetime.fromtimestamp(each_time))
emfitqs_concat.to_csv(path + subject_folder + '_emfitqs.csv')

# Ticwatch
ticwatch_list_df = []
ticwatch_list_filename = glob.glob('./' + subject_folder + '*/Ticwatch/subject*.csv')
if len(ticwatch_list_filename) != 0:
    for fn in ticwatch_list_filename:
        ticwatch_list_df.append(pd.read_csv(fn))
        
    if len(ticwatch_list_df) > 1:
        ticwatch_concat = pd.concat(ticwatch_list_df, ignore_index = True)
    else:
        ticwatch_concat = ticwatch_list_df[0]
    ticwatch_concat.pop('end')
    ticwatch_concat = ticwatch_concat.sort_values(by=['start'])
    ticwatch_concat.rename(columns={'start':'Timestamp', 'value':'HR_ticwatch'}, inplace=True)
    ticwatch_concat['Timestamp'] = ticwatch_concat['Timestamp']/1000
    ticwatch_concat['Timestamp'] = ticwatch_concat['Timestamp'].apply(lambda each_time : dt.datetime.fromtimestamp(each_time))
    ticwatch_concat.to_csv(path + subject_folder + '_ticwatch.csv')

# PolarH10
temp_time = []
polarh10_list_df = []
polarh10_list_filename = glob.glob('./'  + subject_folder + '*/PolarH10/subject*.csv')
for fn in polarh10_list_filename:
    #Processing Timestamp
    polarh10_description_df = pd.read_csv(fn, nrows=1)
    date = polarh10_description_df['Date'][0]
    start_time = polarh10_description_df['Start time'][0]
    polarh10_data_df = pd.read_csv(fn, skiprows=2)
    start_time_obj = dt.datetime.strptime(date + '_' + start_time, '%d-%m-%Y_%H:%M:%S')
    polarh10_data_df = polarh10_data_df.loc[:, ['Time', 'HR (bpm)']]
    each_time_obj = start_time_obj
    for i in range(0, len(polarh10_data_df)):
        each_time_obj = each_time_obj + dt.timedelta(seconds=1)
        temp_time.append(each_time_obj)
        
    polarh10_data_df['Timestamp'] = temp_time
    temp_time = []
    
    polarh10_list_df.append(polarh10_data_df)

if len(polarh10_data_df) > 1 :
    polarh10_concat = pd.concat(polarh10_list_df, ignore_index=True)
else:
    polarh10_concat = polarh10_list_df[0]
polarh10_concat.pop('Time')
polarh10_concat = polarh10_concat.loc[:, ['Timestamp', 'HR (bpm)']]
polarh10_concat.rename(columns={'HR (bpm)':'HR_polarh10'}, inplace=True)
polarh10_concat = polarh10_concat.sort_values(by=('Timestamp'), ascending=True).reset_index(drop=True)
polarh10_concat.to_csv(path + subject_folder + '_polarh10.csv')

# Fitbit
fitbit_df = pd.read_csv(glob.glob('./'  + subject_folder + '*/Fitbit/subject*.csv')[0])
fitbit_df.rename(columns={'time':'Timestamp', 'value':'HR_fitbit'}, inplace=True)
fitbit_df.pop(fitbit_df.columns[0])
fitbit_df['Timestamp'] = fitbit_df['Timestamp'].apply(lambda each_time : dt.datetime.strptime(date + '_' + each_time, '%d-%m-%Y_%H:%M:%S'))
fitbit_df.to_csv(path + subject_folder + '_fitbit.csv')

# AppleWatch4
applewatch_df = pd.read_csv(glob.glob('./' + subject_folder + '*/Apple*/subject*.csv')[0])
applewatch_df.rename(columns={'time': 'Timestamp', 'hr':'HR_applewatch'}, inplace=True)
applewatch_df.pop('date')
applewatch_df.pop('timezone')
applewatch_df['Timestamp'] = applewatch_df['Timestamp'].apply(lambda each_time : dt.datetime.strptime(date + '_' + each_time, '%d-%m-%Y_%H:%M:%S'))
applewatch_df.to_csv(path + subject_folder + '_applewatch.csv')

