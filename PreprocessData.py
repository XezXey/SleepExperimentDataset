#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:23:21 2019
Preprocess the device data file
@author: puntawat
"""
# Import libraries
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

subject_folder = sys.argv[1]

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
    
empatica_merged_df = empatica_dict_df['Bvp']
empatica_merge_column = empatica_filename.remove('Bvp')
for feature in empatica_filename:
    empatica_merged_df = empatica_merged_df.merge(empatica_dict_df[feature], on='TS_Machine', how='outer').sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)

if empatica_merged_df['TS_Machine'] is 
empatica_merged_df['TS_Machine'] = empatica_merged_df['TS_Machine'].apply(lambda each_time : dt.datetime.fromtimestamp(each_time))
empatica_merged_df.rename(columns={'TS_Machine':'Timestamp'}, inplace=True)
empatica_merged_df.to_csv(empatica_list_filename[0] + '.csv')

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

emfitqs_concat = emfitqs_concat.rename(columns={'timestamp_from_machine':'Timestamp'})
emfitqs_concat = emfitqs_concat.sort_values(by=('Timestamp'), ascending=True).reset_index(drop=True)
emfitqs_concat['Timestamp'] = emfitqs_concat['Timestamp'].apply(lambda each_time : dt.datetime.fromtimestamp(each_time))
emfitqs_concat.to_csv(emfitqs_list_filename[0] + '.csv')

# Ticwatch
ticwatch_list_df = []
ticwatch_list_filename = glob.glob('./' + subject_folder + '*/Ticwatch/subject*.csv')
for fn in ticwatch_list_filename:
    ticwatch_list_df.append(pd.read_csv(fn))
    
if len(ticwatch_list_df) > 1:
    ticwatch_concat = pd.concat(ticwatch_list_df, ignore_index = True)
else:
    ticwatch_concat = ticwatch_list_df[0]
ticwatch_concat.pop('end')
ticwatch_concat = ticwatch_concat.sort_values(by=['start'])
ticwatch_concat.rename(columns={'start':'Timestamp', 'value':'Hr'}, inplace=True)
ticwatch_concat['Timestamp'] = ticwatch_concat['Timestamp']/1000
ticwatch_concat['Timestamp'] = ticwatch_concat['Timestamp'].apply(lambda each_time : dt.datetime.fromtimestamp(each_time))
ticwatch_concat.to_csv(ticwatch_list_filename[0] + '.csv')

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
polarh10_concat = polarh10_concat.sort_values(by=('Timestamp'), ascending=True).reset_index(drop=True)
polarh10_concat.to_csv(polarh10_list_filename[0] + '.csv')