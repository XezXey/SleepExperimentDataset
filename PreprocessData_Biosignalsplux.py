#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:23:21 2019
Preprocess the device data file
@author: puntawat
"""
# Import libraries
import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from biosppy.signals import bvp, ecg
import json


#if len(sys.argv) == 1:
#    sys.exit("No subject input")
#subject_folder = sys.argv[1]
all_hr_len = []
sj_names = ['Subject04', 'Subject05', 'Subject06', 'Subject07']
for each_sj in sj_names:
    subject_folder = each_sj
    subject_folder = glob.glob(subject_folder + '*')[0]
    #if subject_folder == []:
    #    sys.exit("Cannot find that subject")
    
    #subject_folder = 'Subject01_2019-1-16'
    
    path = './' + subject_folder + '/' + 'All_Device_Preprocess/'
    # Trying to make directory if it's not exist
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: #Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    # Biosignalsplux
    biosignalsplux_list_filename = glob.glob('./'  + subject_folder + '/Biosignalsplux/subject*.csv')
    all_len = 0
    biosignalsplux_list_df = []
    biosignalsplux_list_hr_df = []
    
    for fn in biosignalsplux_list_filename:
        # Reading file to dataframe
        if each_sj in ['Subject01', 'Subject02', 'Subject03']:
            biosignalsplux_df = pd.read_csv(fn, index_col=None, header=0)
        else:
            # Reading and parse to json
            file_description = open(fn)
            file_description = file_description.readline()
            #print(file_description)
            file_description = file_description.replace('\'', '\"') # json use only " (double quotes)
            file_description = json.loads(file_description)
            
            time = file_description[str(list(file_description.keys())[0])]['time']
            date = file_description[str(list(file_description.keys())[0])]['date']
            start_time = date + '_' + time
            biosignalsplux_df = pd.read_csv(fn, index_col=None, header=1)
        
        start_time_obj = dt.datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S.%f')

            
        # Stamp time to the heart rate record
        if 'ECG' in biosignalsplux_df.columns:
            ecg_data = ecg.ecg(signal=biosignalsplux_df['ECG'], sampling_rate=300, show=True)
            biosignalsplux_hr = pd.DataFrame({'Timestamp' : ecg_data['heart_rate_ts'], 'HR_biosignalsplux' : ecg_data['heart_rate']})
            biosignalsplux_hr['Timestamp'] = biosignalsplux_hr['Timestamp'].apply(lambda each_hr_ts : start_time_obj + dt.timedelta(seconds = each_hr_ts))
            all_len = all_len + len(biosignalsplux_hr['HR_biosignalsplux'])
            biosignalsplux_list_hr_df.append(biosignalsplux_hr)

        elif 'CH3_ECG' in biosignalsplux_df.columns:
            ecg_data = ecg.ecg(signal=biosignalsplux_df['CH3_ECG'], sampling_rate=600, show=True)
            biosignalsplux_hr = pd.DataFrame({'Timestamp' : ecg_data['heart_rate_ts'], 'HR_biosignalsplux' : ecg_data['heart_rate']})
            biosignalsplux_hr['Timestamp'] = biosignalsplux_hr['Timestamp'].apply(lambda each_hr_ts : start_time_obj + dt.timedelta(seconds = each_hr_ts))
            biosignalsplux_list_hr_df.append(biosignalsplux_hr)

        biosignalsplux_hr_concat = pd.concat(biosignalsplux_list_hr_df, ignore_index=True)
        biosignalsplux_hr_concat = biosignalsplux_hr_concat.sort_values(by=('Timestamp'), ascending=True).reset_index(drop=True)
        biosignalsplux_hr_concat.to_csv(path + subject_folder + '_biosignalsplux.csv')

