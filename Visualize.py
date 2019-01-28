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
subject_folder = "Subject01"
"""
if len(sys.argv) == 1:
    sys.exit("No subject input")
"""
print("Data from : " + subject_folder)
path = './' + subject_folder + '*/All_Device_Preprocess/*.csv'


devices_filename = glob.glob(path)
devices_list = ['applewatch', 'fitbit', 'emfitqs', 'empatica', 'polarh10']
devices_dict_df = {}
devices_list_df = []
for index, filename in enumerate(devices_filename):
    devices_list_df.append(pd.read_csv(filename, index_col=0))
    devices_dict_df[devices_list[index]] = pd.read_csv(filename, index_col=0)
    
for each_device in devices_list:
    print(devices_dict_df[each_device].head(3))

devices_df = pd.concat(devices_list_df, ignore_index=True)
#empatica_dict_df[fn].sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)
devices_df = devices_df.loc[:, ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_fitbit', 'HR_emfitqs']].groupby(devices_df['Timestamp']).mean()
plt.plot(devices_df[:10])
plt.legend()
# Groupby Timestamp
# pd.groupby(devices_dict_df['applewatch'],)