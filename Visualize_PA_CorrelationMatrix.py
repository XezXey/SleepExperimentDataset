#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 23:08:02 2019

@author: puntawat
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns; sns.set(color_codes=True)
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import errno


subject_folder = sys.argv[1]
#subject_folder = "Subject01"

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
devices_df = devices_df.loc[:, ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'HR_biosignalsplux', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica']].groupby(devices_df['Timestamp']).mean()
devices_df['PA_lvl_VectorA_empatica_encoded'] = np.ceil(devices_df['PA_lvl_VectorA_empatica_encoded'])

devices_df['Timestamp'] = devices_df.index
devices_df['PA_lvl_VectorA_empatica'] = devices_df['PA_lvl_VectorA_empatica_encoded'].map({1:'Sedentary', 2:'Light', 3:'Moderate', 4:'Vigorous', 5:'Very Vigorous'})

#devices_df['PA_lvl_empatica'] = devices_df['PA_lvl_empatica_encoded'].map({1:'Sedentary', 2:'Light', 3:'Moderate', 4:'Vigorous', 5:'Very Vigorous'})

# Plotting
"""
# Slicing into the interest interval
start_time = "13:00:10"
end_time = "16:38:23"
start_time_obj = dt.datetime.strptime(start_time, "%H:%M:%S").replace(microsecond=0).time()
end_time_obj = dt.datetime.strptime(end_time, "%H:%M:%S").replace(microsecond=0).time()
devices_df_interval = devices_df.loc[(devices_df['Timestamp'] > start_time_obj) & (devices_df['Timestamp'] < end_time_obj)]
"""
# Slicing into the resting and sleeping state
start_time_resting = devices_df['HR_biosignalsplux'].dropna().index[0]
end_time_resting = start_time_resting + dt.timedelta(minutes=30)
start_time_sleeping = end_time_resting + dt.timedelta(minutes=5)
end_time_sleeping = devices_df['HR_biosignalsplux'].dropna().index[-1]
start_time_activity = devices_df['HR_polarh10'].dropna().index[0]
end_time_activity = devices_df['HR_polarh10'].dropna().index[-1]


# For analyze
devices_df_interval_resting = devices_df.loc[(devices_df['Timestamp'] > start_time_resting) & (devices_df['Timestamp'] < end_time_resting)]
devices_df_interval_sleeping = devices_df.loc[(devices_df['Timestamp'] > start_time_sleeping) & (devices_df['Timestamp'] < end_time_sleeping)]
devices_df_interval_activity = devices_df.loc[(devices_df['Timestamp'] > start_time_activity) & (devices_df['Timestamp'] < end_time_activity)]

# Calculate Correlation Matrix compare with biosignalsplux
# Loop over each state using 5 minutes windows gap.
windows_gap = dt.timedelta(minutes=5)
# 1. Resting state devices = HR_empatica, HR_fitbit, HR_biosignalsplux
n_windows_resting = math.ceil((len(devices_df_interval_resting)/60)/5) # Find number of time that use to iterate
resting_interest_columns = ['Timestamp', 'HR_biosignalsplux', 'HR_empatica', 'HR_fitbit', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'PA_lvl_VectorA_empatica', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica']
resting_compare_columns = ['HR_empatica', 'HR_fitbit']
devices_df_interval_resting = devices_df_interval_resting[resting_interest_columns]
corr_resting = {'start_time_window' : [], 'end_time_window' : [], 'n_window' : []}
for each_compare in resting_compare_columns:
    corr_resting['corr_' + each_compare] = []
    corr_resting['n_data_used_' + each_compare] = []
    corr_resting['p_score_' + each_compare] = []
    corr_resting['t_stat_' + each_compare] = []
    corr_resting['std_' + each_compare] = []
    corr_resting['mean_' + each_compare] = []
    corr_resting['mse_' + each_compare] = []
    corr_resting['rmse_' + each_compare] = []
    corr_resting['sem_' + each_compare] = []
    corr_resting['corr_VectorA_' + each_compare] = []

    
for window in range(0, n_windows_resting):
    start_window = start_time_resting + (windows_gap * window)
    end_window = start_time_resting + (windows_gap * (window+1))
    each_window_resting = devices_df_interval_resting.loc[(devices_df_interval_resting['Timestamp'] > (start_time_resting + (windows_gap * window))) & (devices_df_interval_resting['Timestamp'] < (start_time_resting + (windows_gap * (window+1))))]
    corr_resting['n_window'].append(window+1)
    corr_resting['start_time_window'].append(start_window)
    corr_resting['end_time_window'].append(end_window)
    #print("Start time : " + str(each_window_resting.head(1)['Timestamp'].values))
    #print("End time : " + str(each_window_resting.tail(1)['Timestamp'].values))
    # Finding MAE for each pair
    for each_compare in resting_compare_columns:
        resting_cmp_df = each_window_resting[['HR_biosignalsplux', each_compare, 'VectorA_empatica']]
        resting_cmp_df = resting_cmp_df.dropna()
        corr_resting['n_data_used_' + each_compare].append(len(resting_cmp_df))
        corr_resting['corr_' + each_compare].append(resting_cmp_df['HR_biosignalsplux'].corr(resting_cmp_df[each_compare]))
        t_stat, p_score = stats.ttest_ind(resting_cmp_df['HR_biosignalsplux'], resting_cmp_df[each_compare], equal_var = False)
        corr_resting['p_score_' + each_compare].append(p_score)
        corr_resting['t_stat_' + each_compare].append(t_stat)
        corr_resting['mean_' + each_compare].append(np.mean(resting_cmp_df[each_compare]))
        corr_resting['std_' + each_compare].append(np.std(resting_cmp_df[each_compare]))
        corr_resting['sem_' + each_compare].append(stats.sem(resting_cmp_df['HR_biosignalsplux']-resting_cmp_df[each_compare]))
        corr_resting['corr_VectorA_' + each_compare].append(resting_cmp_df['VectorA_empatica'].corr(resting_cmp_df['HR_biosignalsplux']-resting_cmp_df[each_compare]))
        try:
            corr_resting['mse_' + each_compare].append(mean_squared_error(y_true=resting_cmp_df['HR_biosignalsplux'], y_pred=resting_cmp_df[each_compare]))
            corr_resting['rmse_' + each_compare].append(math.sqrt(mean_squared_error(y_true=resting_cmp_df['HR_biosignalsplux'], y_pred=resting_cmp_df[each_compare])))
        except ValueError : 
            print("No matching data at a time")
            corr_resting['mse_' + each_compare].append(np.nan)
            corr_resting['rmse_' + each_compare].append(np.nan)
        
        
# 2. Sleeping state devices = HR_empatica, HR_fitbit, HR_biosignalsplux, 'HR_emfitqs
n_windows_sleeping = math.ceil((len(devices_df_interval_sleeping)/60)/5) # Find number of time that use to iterate
sleeping_interest_columns = ['Timestamp', 'HR_biosignalsplux', 'HR_empatica', 'HR_fitbit', 'HR_emfitqs', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'PA_lvl_VectorA_empatica', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica']
sleeping_compare_columns = ['HR_empatica', 'HR_fitbit', 'HR_emfitqs']
devices_df_interval_sleeping = devices_df_interval_sleeping[sleeping_interest_columns]
corr_sleeping = {'start_time_window' : [], 'end_time_window' : [], 'n_window' : []}
for each_compare in sleeping_compare_columns:
    corr_sleeping['corr_' + each_compare] = []
    corr_sleeping['n_data_used_' + each_compare] = []
    corr_sleeping['p_score_' + each_compare] = []
    corr_sleeping['t_stat_' + each_compare] = []
    corr_sleeping['std_' + each_compare] = []
    corr_sleeping['mean_' + each_compare] = []    
    corr_sleeping['mse_' + each_compare] = []
    corr_sleeping['rmse_' + each_compare] = []
    corr_sleeping['sem_' + each_compare] = []
    corr_sleeping['corr_VectorA_' + each_compare] = []



for window in range(0, n_windows_sleeping):
    start_window = start_time_sleeping + (windows_gap * window)
    end_window = start_time_sleeping + (windows_gap * (window+1))
    each_window_sleeping = devices_df_interval_sleeping.loc[(devices_df_interval_sleeping['Timestamp'] > start_window) & (devices_df_interval_sleeping['Timestamp'] < end_window)]
    corr_sleeping['n_window'].append(window+1)
    corr_sleeping['start_time_window'].append(start_window)
    corr_sleeping['end_time_window'].append(end_window)
    #print("Start time : " + str(each_window_resting.head(1)['Timestamp'].values))
    #print("End time : " + str(each_window_resting.tail(1)['Timestamp'].values))
    for each_compare in sleeping_compare_columns:
        sleeping_cmp_df = each_window_sleeping[['HR_biosignalsplux', each_compare, 'VectorA_empatica']]
        sleeping_cmp_df = sleeping_cmp_df.dropna()
        corr_sleeping['corr_' + each_compare].append(sleeping_cmp_df['HR_biosignalsplux'].corr(sleeping_cmp_df[each_compare]))
        corr_sleeping['n_data_used_' + each_compare].append(len(sleeping_cmp_df))
        t_stat, p_score = stats.ttest_ind(sleeping_cmp_df['HR_biosignalsplux'], sleeping_cmp_df[each_compare], equal_var = False)
        corr_sleeping['p_score_' + each_compare].append(p_score)
        corr_sleeping['t_stat_' + each_compare].append(t_stat)
        corr_sleeping['mean_' + each_compare].append(np.mean(sleeping_cmp_df[each_compare]))
        corr_sleeping['std_' + each_compare].append(np.std(sleeping_cmp_df[each_compare]))
        corr_sleeping['sem_' + each_compare].append(stats.sem(sleeping_cmp_df['HR_biosignalsplux']-sleeping_cmp_df[each_compare]))
        try:
            corr_sleeping['mse_' + each_compare].append(mean_squared_error(y_true=sleeping_cmp_df['HR_biosignalsplux'], y_pred=sleeping_cmp_df[each_compare]))
            corr_sleeping['rmse_' + each_compare].append(math.sqrt(mean_squared_error(y_true=sleeping_cmp_df['HR_biosignalsplux'], y_pred=sleeping_cmp_df[each_compare])))
        except ValueError : 
            print("No matching data at a time")
            corr_sleeping['mse_' + each_compare].append(np.nan)
            corr_sleeping['rmse_' + each_compare].append(np.nan)
        corr_sleeping['corr_VectorA_' + each_compare].append(sleeping_cmp_df['VectorA_empatica'].corr(sleeping_cmp_df['HR_biosignalsplux']-sleeping_cmp_df[each_compare]))

            
        

        
# 3. Activity state devices = HR_empatica, HR_fitbit, HR_biosignalsplux, 'HR_emfitqs
n_windows_activity = math.ceil((len(devices_df_interval_activity)/60)/5) # Find number of time that use to iterate
activity_interest_columns = ['Timestamp', 'HR_polarh10', 'HR_fitbit', 'HR_empatica', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'PA_lvl_VectorA_empatica', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica']
activity_compare_columns = ['HR_empatica', 'HR_fitbit']
devices_df_interval_activity = devices_df_interval_activity[activity_interest_columns]
corr_activity = {'start_time_window' : [], 'end_time_window' : [], 'n_window' : []}
for each_compare in activity_compare_columns:
    corr_activity['corr_' + each_compare] = []
    corr_activity['n_data_used_' + each_compare] = []
    corr_activity['p_score_' + each_compare] = []
    corr_activity['t_stat_' + each_compare] = []
    corr_activity['std_' + each_compare] = []
    corr_activity['mean_' + each_compare] = []    
    corr_activity['mse_' + each_compare] = []
    corr_activity['rmse_' + each_compare] = []
    corr_activity['sem_' + each_compare] = []
    corr_activity['corr_VectorA_' + each_compare] = []
    
for window in range(0, n_windows_activity):
    start_window = start_time_activity + (windows_gap * window)
    end_window = start_time_activity + (windows_gap * (window+1))
    each_window_activity = devices_df_interval_activity.loc[(devices_df_interval_activity['Timestamp'] > start_window) & (devices_df_interval_activity['Timestamp'] < end_window)]
    corr_activity['n_window'].append(window+1)
    corr_activity['start_time_window'].append(start_window)
    corr_activity['end_time_window'].append(end_window)
    #print("Start time : " + str(each_window_resting.head(1)['Timestamp'].values))
    #print("End time : " + str(each_window_resting.tail(1)['Timestamp'].values))
    
    for each_compare in activity_compare_columns:
        activity_cmp_df = each_window_activity[['HR_polarh10', each_compare, 'VectorA_empatica']]
        activity_cmp_df = activity_cmp_df.dropna()
        corr_activity['corr_' + each_compare].append(activity_cmp_df['HR_polarh10'].corr(activity_cmp_df[each_compare]))
        corr_activity['n_data_used_' + each_compare].append(len(activity_cmp_df))
        t_stat, p_score = stats.ttest_ind(activity_cmp_df['HR_polarh10'], activity_cmp_df[each_compare], equal_var = False)
        corr_activity['p_score_' + each_compare].append(p_score)
        corr_activity['t_stat_' + each_compare].append(t_stat)
        corr_activity['std_' + each_compare].append(np.std(activity_cmp_df[each_compare]))
        corr_activity['mean_' + each_compare].append(np.mean(activity_cmp_df[each_compare]))
        corr_activity['sem_' + each_compare].append(stats.sem(activity_cmp_df['HR_polarh10']-activity_cmp_df[each_compare]))
        corr_activity['corr_VectorA_' + each_compare].append(activity_cmp_df['VectorA_empatica'].corr(activity_cmp_df['HR_polarh10']-activity_cmp_df[each_compare]))
        try:
            corr_activity['mse_' + each_compare].append(mean_squared_error(y_true=activity_cmp_df['HR_polarh10'], y_pred=activity_cmp_df[each_compare]))
            corr_activity['rmse_' + each_compare].append(math.sqrt(mean_squared_error(y_true=activity_cmp_df['HR_polarh10'], y_pred=activity_cmp_df[each_compare])))
        except ValueError : 
            print("No matching data at a time")
            corr_activity['mse_' + each_compare].append(np.nan)
            corr_activity['rmse_' + each_compare].append(np.nan)


corr_resting_df = pd.DataFrame(corr_resting)
corr_sleeping_df = pd.DataFrame(corr_sleeping)
corr_activity_df = pd.DataFrame(corr_activity)


# Plotting

# Create folder to store image
path_img = './Visualization_Image/Correlation/' + subject_folder + '/'
# Trying to make directory if it's not exist
if not os.path.exists(os.path.dirname(path_img)):
    try:
        os.makedirs(os.path.dirname(path_img))
    except OSError as exc: #Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


# 1. Correlation coefficient
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
fig.suptitle(subject_folder + ' - Resting state(Correlation Coefficient with Biosignalsplux)', fontsize=40)
axes_corr_resting = plt.subplot(2, 2, 1)
axes_corr_resting.plot(corr_resting_df['n_window'], corr_resting_df['corr_HR_fitbit'], 'o-')
axes_corr_resting.set(ylabel='Fitbit - Correlation Coefficient with Biosignalsplux', xlabel='n_windows')

axes_corr_resting = plt.subplot(2, 2, 2)
axes_corr_resting.plot(corr_resting_df['n_window'], corr_resting_df['corr_HR_empatica'], 'x-')
axes_corr_resting.set(ylabel='Empatica - Correlation Coefficient with Biosignalsplux', xlabel='n_windows')

axes_corr_resting = plt.subplot(2, 2, 3)
axes_corr_resting.plot(corr_resting_df['n_window'], corr_resting_df['corr_HR_fitbit'], 'o-')
axes_corr_resting.plot(corr_resting_df['n_window'], corr_resting_df['corr_HR_empatica'], 'x-')
axes_corr_resting.set(ylabel='Correlation Coefficient with Biosignalsplux', xlabel='n_windows')
plt.legend()

fig.savefig(path_img + subject_folder + '_corr_resting_state', quality=95)


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
fig.suptitle(subject_folder + ' - Sleeping state(Correlation Coefficient with Biosignalsplux)', fontsize=40)
axes_corr_sleeping = plt.subplot(2, 2, 1)
axes_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_HR_fitbit'], 'o-')
axes_corr_sleeping.set(ylabel='Fitbit - Correlation Coefficient with Biosignalsplux', xlabel='n_windows')

axes_corr_sleeping = plt.subplot(2, 2, 2)
axes_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_HR_empatica'], 'x-')
axes_corr_sleeping.set(ylabel='Empatica - Correlation Coefficient with Biosignalsplux', xlabel='n_windows')

axes_corr_sleeping = plt.subplot(2, 2, 3)
axes_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_HR_emfitqs'], '*-')
axes_corr_sleeping.set(ylabel='EmfitQS - Correlation Coefficient with Biosignalsplux', xlabel='n_windows')

axes_corr_sleeping = plt.subplot(2, 2, 4)
axes_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_HR_fitbit'], 'o-')
axes_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_HR_empatica'], 'x-')
axes_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_HR_emfitqs'], '*-')
axes_corr_sleeping.set(ylabel='Correlation Coefficient with Biosignalsplux', xlabel='n_windows')
plt.legend()

fig.savefig(path_img + subject_folder + '_corr_sleeping_state', quality=95)


# Activity state
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Activity state(Correlation Coefficient with PolarH10)', fontsize=40)
axes_corr_activity = plt.subplot(2, 2, 1)
axes_corr_activity.plot(corr_activity_df['n_window'], corr_activity_df['corr_HR_fitbit'], 'o-')
axes_corr_activity.set(ylabel='Fitbit - Correlation Coefficient with PolarH10', xlabel='n_windows')

axes_corr_activity = plt.subplot(2, 2, 2)
axes_corr_activity.plot(corr_activity_df['n_window'], corr_activity_df['corr_HR_empatica'], 'x-')
axes_corr_activity.set(ylabel='Empatica - Correlation Coefficient with PolarH10', xlabel='n_windows')

axes_corr_activity = plt.subplot(2, 2, 3)
axes_corr_activity.plot(corr_activity_df['n_window'], corr_activity_df['corr_HR_fitbit'], 'o-')
axes_corr_activity.plot(corr_activity_df['n_window'], corr_activity_df['corr_HR_empatica'], 'x-')
axes_corr_activity.set(ylabel='Correlation Coefficient with Biosignalsplux', xlabel='n_windows')
plt.legend()
fig.savefig(path_img + subject_folder + '_corr_activity_state', quality=95)




# 2. Standard error plot
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
fig.suptitle(subject_folder + ' - Resting state(Standard Error)', fontsize=40)
axes_stderr_resting = plt.subplot(2, 2, 1)
axes_stderr_resting.errorbar(corr_resting_df['n_window'], corr_resting_df['mean_HR_fitbit'], yerr=corr_resting_df['rmse_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_resting.set(ylabel='Fitbit - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_resting = plt.subplot(2, 2, 2)
axes_stderr_resting.errorbar(corr_resting_df['n_window'], corr_resting_df['mean_HR_empatica'], yerr=corr_resting_df['rmse_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_resting.set(ylabel='Empatica - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_resting = plt.subplot(2, 2, 3)
axes_stderr_resting.errorbar(corr_resting_df['n_window'], corr_resting_df['mean_HR_fitbit'], yerr=corr_resting_df['rmse_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_resting.errorbar(corr_resting_df['n_window'], corr_resting_df['mean_HR_empatica'], yerr=corr_resting_df['rmse_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_resting.set(ylabel='Heart rate(bpm)', xlabel='n_windows')
plt.legend()
fig.savefig(path_img + subject_folder + '_stderr_resting_state', quality=95)


# 2. Standard error plot
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
fig.suptitle(subject_folder + ' - Sleeping state(Standard Error)', fontsize=40)
axes_stderr_sleeping = plt.subplot(2, 2, 1)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_fitbit'], yerr=corr_sleeping_df['rmse_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.set(ylabel='Fitbit - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_sleeping = plt.subplot(2, 2, 2)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_empatica'], yerr=corr_sleeping_df['rmse_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.set(ylabel='Empatica - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_sleeping = plt.subplot(2, 2, 3)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_emfitqs'], yerr=corr_sleeping_df['rmse_HR_emfitqs'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.set(ylabel='EmfitQS - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_sleeping = plt.subplot(2, 2, 4)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_fitbit'], yerr=corr_sleeping_df['rmse_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_empatica'], yerr=corr_sleeping_df['rmse_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_emfitqs'], yerr=corr_sleeping_df['rmse_HR_emfitqs'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.set(ylabel='Heart rate(bpm)', xlabel='n_windows')
plt.legend()
fig.savefig(path_img + subject_folder + '_stderr_sleeping_state', quality=95)


# Activity state
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Activity state(Standard Error)', fontsize=40)
axes_stderr_activity = plt.subplot(2, 2, 1)
axes_stderr_activity.errorbar(corr_activity_df['n_window'], corr_activity_df['mean_HR_fitbit'], yerr=corr_activity_df['rmse_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_activity.set(ylabel='Fitbit - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_activity = plt.subplot(2, 2, 2)
axes_stderr_activity.errorbar(corr_activity_df['n_window'], corr_activity_df['mean_HR_empatica'], yerr=corr_activity_df['rmse_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_activity.set(ylabel='Empatica - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_activity = plt.subplot(2, 2, 3)
axes_stderr_activity.errorbar(corr_activity_df['n_window'], corr_activity_df['mean_HR_fitbit'], yerr=corr_activity_df['rmse_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_activity.errorbar(corr_activity_df['n_window'], corr_activity_df['mean_HR_empatica'], yerr=corr_activity_df['rmse_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_activity.set(ylabel='Heart rate(bpm)', xlabel='n_windows')
plt.legend()
fig.savefig(path_img + subject_folder + '_stderr_activity_state', quality=95)


# 3. Error correlation with Acc
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
fig.suptitle(subject_folder + ' - Resting state(Correlation Coefficient with ACC)', fontsize=40)
axes_acc_corr_resting = plt.subplot(2, 2, 1)
axes_acc_corr_resting.plot(corr_resting_df['n_window'], corr_resting_df['corr_VectorA_HR_empatica'], 'o-')
axes_acc_corr_resting.set(ylabel='Empatica - Correlation Coefficient with ACC', xlabel='n_windows')


axes_acc_corr_resting = plt.subplot(2, 2, 2)
axes_acc_corr_resting.plot(corr_resting_df['n_window'], corr_resting_df['corr_VectorA_HR_fitbit'], 'x-')
axes_acc_corr_resting.set(ylabel='Fitbit - Correlation Coefficient with ACC', xlabel='n_windows')

axes_acc_corr_resting = plt.subplot(2, 2, 3)
axes_acc_corr_resting.plot(corr_resting_df['n_window'], corr_resting_df['corr_VectorA_HR_fitbit'], 'x-')
axes_acc_corr_resting.plot(corr_resting_df['n_window'], corr_resting_df['corr_VectorA_HR_empatica'], 'o-')
axes_acc_corr_resting.set(ylabel='Correlation Coefficient with ACC', xlabel='n_windows')
plt.legend()
fig.savefig(path_img + subject_folder + '_corr_acc_resting_state', quality=95)

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
fig.suptitle(subject_folder + ' - Sleeping state(Correlation Coefficient with ACC)', fontsize=40)
axes_acc_corr_sleeping = plt.subplot(2, 2, 1)
axes_acc_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_VectorA_HR_empatica'], 'o-')
axes_acc_corr_sleeping.set(ylabel='Empatica - Correlation Coefficient with ACC', xlabel='n_windows')

axes_acc_corr_sleeping = plt.subplot(2, 2, 2)
axes_acc_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_VectorA_HR_fitbit'], 'x-')
axes_acc_corr_sleeping.set(ylabel='Fitbit - Correlation Coefficient with ACC', xlabel='n_windows')

axes_acc_corr_sleeping = plt.subplot(2, 2, 3)
axes_acc_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_VectorA_HR_emfitqs'], 'x-')
axes_acc_corr_sleeping.set(ylabel='EmfitQS - Correlation Coefficient with ACC', xlabel='n_windows')


axes_acc_corr_sleeping = plt.subplot(2, 2, 4)
axes_acc_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_VectorA_HR_fitbit'], 'x-')
axes_acc_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_VectorA_HR_empatica'], 'o-')
axes_acc_corr_sleeping.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_VectorA_HR_emfitqs'], '*-')
axes_acc_corr_sleeping.set(ylabel='Correlation Coefficient with ACC', xlabel='n_windows')
plt.legend()
fig.savefig(path_img + subject_folder + '_corr_acc_sleeping_state', quality=95)

# Activity state
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Activity state(Correlation Coefficient with ACC)', fontsize=40)
axes_acc_corr_activity = plt.subplot(2, 2, 1)
axes_acc_corr_activity.plot(corr_activity_df['n_window'], corr_activity_df['corr_VectorA_HR_empatica'], 'o-')
axes_acc_corr_activity.set(ylabel='Empatica - Correlation Coefficient with ACC', xlabel='n_windows')


axes_acc_corr_activity = plt.subplot(2, 2, 2)
axes_acc_corr_activity.plot(corr_activity_df['n_window'], corr_activity_df['corr_VectorA_HR_fitbit'], 'x-')
axes_acc_corr_activity.set(ylabel='Fitbit - Correlation Coefficient with ACC', xlabel='n_windows')

axes_acc_corr_activity = plt.subplot(2, 2, 3)
axes_acc_corr_activity.plot(corr_activity_df['n_window'], corr_activity_df['corr_VectorA_HR_fitbit'], 'x-')
axes_acc_corr_activity.plot(corr_activity_df['n_window'], corr_activity_df['corr_VectorA_HR_empatica'], 'o-')
axes_acc_corr_activity.set(ylabel='Correlation Coefficient with ACC', xlabel='n_windows')
plt.legend()
fig.savefig(path_img + subject_folder + '_corr_acc_activity_state', quality=95)



# 4. Freedson with 
"""
x = devices_df_interval_resting.loc[(devices_df_interval_resting['Timestamp'] > start_time_resting.time()) & (devices_df_interval_resting['Timestamp'] < (start_time_resting + windows_gap).time())]
plt.plot(x['PA_lvl_empatica_encoded'])

windows_gap = dt.timedelta(minutes=90)
x = devices_df_interval_sleeping.loc[(devices_df_interval_sleeping['Timestamp'] > start_time_sleeping.time()) & (devices_df_interval_sleeping['Timestamp'] < (start_time_sleeping + windows_gap).time())]
plt.plot(x['PA_lvl_empatica_encoded'])

windows_gap = dt.timedelta(minutes=90)
x = devices_df_interval_activity.loc[(devices_df_interval_activity['Timestamp'] > start_time_activity.time()) & (devices_df_interval_activity['Timestamp'] < (start_time_activity + windows_gap).time())]
plt.plot(x['PA_lvl_empatica_encoded'])
"""