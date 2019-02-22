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


initial_flag_resting = 1
initial_flag_sleeping = 1
initial_flag_activity = 1

#subject_folder = sys.argv[1]
subject_list = ['Subject01', 'Subject02', 'Subject03', 'Subject04', 'Subject05', 'Subject06', 'Subject07', 'Subject08', 'Subject09', 'Subject10']
#subject_folder = "Subject09"

for subject_folder in subject_list:
   
    print("Visualizing CorrelationMatrix all subjects, Data from : " + subject_folder)
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
    devices_df = devices_df.loc[:, ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'HR_biosignalsplux', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica']].groupby(devices_df['Timestamp']).mean()
    devices_df['PA_lvl_VectorA_empatica_encoded'] = np.ceil(devices_df['PA_lvl_VectorA_empatica_encoded'])
    
    devices_df['Timestamp'] = devices_df.index
    devices_df['PA_lvl_VectorA_empatica'] = devices_df['PA_lvl_VectorA_empatica_encoded'].map({1:'Sedentary', 2:'Light', 3:'Moderate', 4:'Vigorous', 5:'Very Vigorous'})
    
    # Filter over 200 bpm out from HR_biosignalsplux
    devices_df['HR_biosignalsplux'] = devices_df['HR_biosignalsplux'].where(devices_df['HR_biosignalsplux'] < 200, np.nan)
    
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
    start_time_resting = devices_df['AX_empatica'].dropna().index[0]
    end_time_resting = start_time_resting + dt.timedelta(minutes=30)
    start_time_sleeping = end_time_resting + dt.timedelta(minutes=5)
    end_time_sleeping = devices_df['AX_empatica'].dropna().index[-1]
    start_time_activity = devices_df['HR_polarh10'].dropna().index[0]
    end_time_activity = devices_df['HR_polarh10'].dropna().index[-1]
    
    
    # For analyze
    devices_df_interval_resting = devices_df.loc[(devices_df['Timestamp'] > start_time_resting.time()) & (devices_df['Timestamp'] < end_time_resting.time())]
    devices_df_interval_sleeping = devices_df.loc[(devices_df['Timestamp'] > start_time_sleeping.time()) & (devices_df['Timestamp'] < end_time_sleeping.time())]
    real_end_of_sleeping_index = devices_df_interval_sleeping['AX_empatica'].dropna().index[-1]
    devices_df_interval_sleeping = devices_df_interval_sleeping.loc[devices_df_interval_sleeping.index < real_end_of_sleeping_index]
    devices_df_interval_activity = devices_df.loc[(devices_df['Timestamp'] > start_time_activity.time()) & (devices_df['Timestamp'] < end_time_activity.time())]

    # Calculate Correlation Matrix compare with biosignalsplux
    # Loop over each state using 5 minutes windows gap.
    windows_gap = dt.timedelta(minutes=5)
    # 1. Resting state devices = HR_empatica, HR_fitbit, HR_biosignalsplux
    n_windows_resting = math.ceil((len(devices_df_interval_resting)/60)/5) # Find number of time that use to iterate
    resting_interest_columns = ['Timestamp', 'HR_biosignalsplux', 'HR_empatica', 'HR_fitbit', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'PA_lvl_VectorA_empatica', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica']
    resting_compare_columns = ['HR_empatica', 'HR_fitbit']
    devices_df_interval_resting = devices_df_interval_resting[resting_interest_columns]
    if initial_flag_resting == 1:
        corr_resting = {'start_time_window' : [], 'end_time_window' : [], 'n_window' : [], 'subject_no' : []}
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
            corr_resting['mae_' + each_compare] = []
        initial_flag_resting = 0
        
    for window in range(0, n_windows_resting):
        start_window = start_time_resting + (windows_gap * window)
        end_window = start_time_resting + (windows_gap * (window+1))
        each_window_resting = devices_df_interval_resting.loc[(devices_df_interval_resting['Timestamp'] > (start_time_resting + (windows_gap * window))) & (devices_df_interval_resting['Timestamp'] < (start_time_resting + (windows_gap * (window+1))))]
        corr_resting['n_window'].append(window+1)
        corr_resting['start_time_window'].append(start_window)
        corr_resting['end_time_window'].append(end_window)
        corr_resting['subject_no'].append(subject_folder)
    
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
                corr_resting['mae_' + each_compare].append(mean_absolute_error(y_true=resting_cmp_df['HR_biosignalsplux'], y_pred=resting_cmp_df[each_compare]))
            except ValueError : 
                print("(Resting)No matching data to compute MSE, RMSE and MAE")
                corr_resting['mse_' + each_compare].append(np.nan)
                corr_resting['rmse_' + each_compare].append(np.nan)
                corr_resting['mae_' + each_compare].append(np.nan)
            
            
    # 2. Sleeping state devices = HR_empatica, HR_fitbit, HR_biosignalsplux, 'HR_emfitqs
    n_windows_sleeping = math.ceil((len(devices_df_interval_sleeping)/60)/5) # Find number of time that use to iterate
    sleeping_interest_columns = ['Timestamp', 'HR_biosignalsplux', 'HR_empatica', 'HR_fitbit', 'HR_emfitqs', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'PA_lvl_VectorA_empatica', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica']
    sleeping_compare_columns = ['HR_empatica', 'HR_fitbit', 'HR_emfitqs']
    devices_df_interval_sleeping = devices_df_interval_sleeping[sleeping_interest_columns]
    
    if initial_flag_sleeping == 1:
        corr_sleeping = {'start_time_window' : [], 'end_time_window' : [], 'n_window' : [], 'subject_no' : []}
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
            corr_sleeping['mae_' + each_compare] = []
        initial_flag_sleeping = 0
    
    
    
    for window in range(0, n_windows_sleeping):
        start_window = start_time_sleeping + (windows_gap * window)
        end_window = start_time_sleeping + (windows_gap * (window+1))
        each_window_sleeping = devices_df_interval_sleeping.loc[(devices_df_interval_sleeping['Timestamp'] > start_window) & (devices_df_interval_sleeping['Timestamp'] < end_window)]
        corr_sleeping['n_window'].append(window+1)
        corr_sleeping['start_time_window'].append(start_window)
        corr_sleeping['end_time_window'].append(end_window)
        corr_sleeping['subject_no'].append(subject_folder)
    
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
            corr_sleeping['corr_VectorA_' + each_compare].append(sleeping_cmp_df['VectorA_empatica'].corr(sleeping_cmp_df['HR_biosignalsplux']-sleeping_cmp_df[each_compare]))
            try:
                corr_sleeping['mse_' + each_compare].append(mean_squared_error(y_true=sleeping_cmp_df['HR_biosignalsplux'], y_pred=sleeping_cmp_df[each_compare]))
                corr_sleeping['rmse_' + each_compare].append(math.sqrt(mean_squared_error(y_true=sleeping_cmp_df['HR_biosignalsplux'], y_pred=sleeping_cmp_df[each_compare])))
                corr_sleeping['mae_' + each_compare].append(mean_absolute_error(y_true=sleeping_cmp_df['HR_biosignalsplux'], y_pred=sleeping_cmp_df[each_compare]))
            except ValueError : 
                print("(Sleeping)No matching data to compute MSE, RMSE and MAE")
                corr_sleeping['mse_' + each_compare].append(np.nan)
                corr_sleeping['rmse_' + each_compare].append(np.nan)
                corr_sleeping['mae_' + each_compare].append(np.nan)
    
                
    
                
        
    # 3. Activity state devices = HR_empatica, HR_fitbit, HR_biosignalsplux, 'HR_emfitqs
    n_windows_activity = math.ceil((len(devices_df_interval_activity)/60)/5) # Find number of time that use to iterate
    activity_interest_columns = ['Timestamp', 'HR_polarh10', 'HR_fitbit', 'HR_empatica', 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'PA_lvl_VectorA_empatica', 'PA_lvl_VectorA_empatica_encoded', 'VectorA_empatica']
    activity_compare_columns = ['HR_empatica', 'HR_fitbit']
    devices_df_interval_activity = devices_df_interval_activity[activity_interest_columns]
    if initial_flag_activity == 1:
        corr_activity = {'start_time_window' : [], 'end_time_window' : [], 'n_window' : [], 'subject_no' : []}
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
            corr_activity['mae_' + each_compare] = []
        initial_flag_activity = 0
        
    for window in range(0, n_windows_activity):
        start_window = start_time_activity + (windows_gap * window)
        end_window = start_time_activity + (windows_gap * (window+1))
        each_window_activity = devices_df_interval_activity.loc[(devices_df_interval_activity['Timestamp'] > start_window) & (devices_df_interval_activity['Timestamp'] < end_window)]
        corr_activity['start_time_window'].append(start_window)
        corr_activity['end_time_window'].append(end_window)
        corr_activity['subject_no'].append(subject_folder)
        corr_activity['n_window'].append(window+1)
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
                corr_activity['mae_' + each_compare].append(mean_absolute_error(y_true=activity_cmp_df['HR_polarh10'], y_pred=activity_cmp_df[each_compare]))
    
            except ValueError : 
                print("(Activity)No matching data to compute MSE, RMSE and MAE")
                corr_activity['mse_' + each_compare].append(np.nan)
                corr_activity['rmse_' + each_compare].append(np.nan)
                corr_activity['mae_' + each_compare].append(np.nan)


# Convert to correlation dict of 3 states to dataframe
corr_resting_df = pd.DataFrame(corr_resting)
corr_sleeping_df = pd.DataFrame(corr_sleeping)
corr_activity_df = pd.DataFrame(corr_activity)
# Filter the interval when there's no devices data out
corr_resting_df = corr_resting_df.loc[(corr_resting_df['n_data_used_HR_empatica']!=0) & (corr_resting_df['n_data_used_HR_fitbit']!=0)]
corr_sleeping_df = corr_sleeping_df.loc[(corr_sleeping_df['n_data_used_HR_empatica']!=0) & (corr_sleeping_df['n_data_used_HR_fitbit']!=0) & (corr_sleeping_df['n_data_used_HR_emfitqs']!=0)]
corr_activity_df = corr_activity_df.loc[(corr_activity_df['n_data_used_HR_empatica']!=0) & (corr_activity_df['n_data_used_HR_fitbit']!=0)]

# Reindex of n_windows
for each_sj in subject_list:
    corr_sleeping_df.loc[corr_sleeping_df['subject_no'] == each_sj, 'n_window'] = list(range(1, len(corr_sleeping_df.loc[corr_sleeping_df['subject_no'] == each_sj]['n_window']) + 1))
    

# Plotting

# Create folder to store image
path_img = './Visualization_Image/Correlation_allsj/'
# Trying to make directory if it's not exist
if not os.path.exists(os.path.dirname(path_img)):
    try:
        os.makedirs(os.path.dirname(path_img))
    except OSError as exc: #Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


# Following the paper plan
""" This will be plot in the 1sj version
# 1. Correlation Coefficient by state
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Device Correlation Coefficient with Biosignalsplux', fontsize=40)
axes_devices_corr = plt.subplot(2, 2, 1)
axes_devices_corr.errorbar(['Resting', 'Sleeping', 'Activity'], [corr_resting_df['corr_HR_fitbit'].mean(), corr_sleeping_df['corr_HR_fitbit'].mean(), corr_activity_df['corr_HR_fitbit'].mean()], 
              yerr=[np.std(corr_resting_df['corr_HR_fitbit']), np.std(corr_sleeping_df['corr_HR_fitbit']), np.std(corr_activity_df['corr_HR_fitbit'])], 
              fmt='x', elinewidth=2.5, markersize=10)
axes_devices_corr.set(ylabel='Fitbit Correlation Coefficient with Biosignalsplux', xlabel='States')


axes_devices_corr = plt.subplot(2, 2, 2)
axes_devices_corr.errorbar(['Resting', 'Sleeping', 'Activity'], [corr_resting_df['corr_HR_empatica'].mean(), corr_sleeping_df['corr_HR_empatica'].mean(), corr_activity_df['corr_HR_empatica'].mean()], 
              yerr=[np.std(corr_resting_df['corr_HR_empatica']), np.std(corr_sleeping_df['corr_HR_empatica']), np.std(corr_activity_df['corr_HR_empatica'])], 
              fmt='o', elinewidth=2.5, markersize=10)
axes_devices_corr.set(ylabel='Empatica Correlation Coefficient with Biosignalsplux', xlabel='States')

axes_devices_corr = plt.subplot(2, 2, 3)
axes_devices_corr.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_HR_emfitqs'], '*')
axes_devices_corr.set(ylabel='EmfitQS - Correlation Coefficient with Biosignalsplux', xlabel='n_windows')

axes_devices_corr = plt.subplot(2, 2, 4)
axes_devices_corr.errorbar(['Resting', 'Sleeping', 'Activity'], [corr_resting_df['corr_HR_fitbit'].mean(), corr_sleeping_df['corr_HR_fitbit'].mean(), corr_activity_df['corr_HR_fitbit'].mean()], 
              yerr=[np.std(corr_resting_df['corr_HR_fitbit']), np.std(corr_sleeping_df['corr_HR_fitbit']), np.std(corr_activity_df['corr_HR_fitbit'])], 
              fmt='x', elinewidth=2.5, markersize=10)
axes_devices_corr.errorbar(['Resting', 'Sleeping', 'Activity'], [corr_resting_df['corr_HR_empatica'].mean(), corr_sleeping_df['corr_HR_empatica'].mean(), corr_activity_df['corr_HR_empatica'].mean()], 
              yerr=[np.std(corr_resting_df['corr_HR_empatica']), np.std(corr_sleeping_df['corr_HR_empatica']), np.std(corr_activity_df['corr_HR_empatica'])], 
              fmt='o', elinewidth=2.5, markersize=10)
axes_devices_corr.set(ylabel='All Devices Correlation Coefficient with Biosignalsplux', xlabel='States')
plt.legend()
"""
# 2. Mean Standard Error using absolute error

# By devices & windows (all subject)
# Grouping by window
corr_resting_groupby_window = corr_resting_df.groupby(corr_resting_df['n_window']).mean().dropna()
corr_sleeping_groupby_window = corr_sleeping_df.groupby(corr_sleeping_df['n_window']).mean().dropna()
corr_activity_groupby_window = corr_activity_df.groupby(corr_activity_df['n_window']).mean().dropna()
corr_all_sj_groupby_window = pd.concat([corr_resting_groupby_window, corr_sleeping_groupby_window, corr_activity_groupby_window], ignore_index = True)

fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle("All devices & All states - Heart rate and error compare with standard", fontsize=40)
axes_corr_all_sj_groupby_window = plt.subplot(2, 2, 1)
axes_corr_all_sj_groupby_window.axvline(x = 5.5, color='r', linestyle='--')
axes_corr_all_sj_groupby_window.text(4.5, 140, 'Resting_to_Sleeping', rotation=90, verticalalignment='center')
axes_corr_all_sj_groupby_window.axvline(x = 22.5, color='r', linestyle='--')
axes_corr_all_sj_groupby_window.text(21.5, 140, 'Sleeping_to_Activity', rotation=90, verticalalignment='center')
axes_corr_all_sj_groupby_window.errorbar(corr_all_sj_groupby_window.index, corr_all_sj_groupby_window['mean_HR_fitbit'], yerr=corr_all_sj_groupby_window['mae_HR_fitbit'], fmt = 'x')
axes_corr_all_sj_groupby_window.set(ylabel='Fitbit - Heart rate(bmp)', xlabel='windows')

axes_corr_all_sj_groupby_window = plt.subplot(2, 2, 2)
axes_corr_all_sj_groupby_window.errorbar(corr_all_sj_groupby_window.index, corr_all_sj_groupby_window['mean_HR_emfitqs'], yerr=corr_all_sj_groupby_window['mae_HR_emfitqs'], fmt = 'x')
axes_corr_all_sj_groupby_window.set(ylabel='Emfitqs - Heart rate(bmp)', xlabel='windows')

axes_corr_all_sj_groupby_window = plt.subplot(2, 2, 3)
axes_corr_all_sj_groupby_window.axvline(x = 5.5, color='r', linestyle='--')
axes_corr_all_sj_groupby_window.text(4.5, 140, 'Resting_to_Sleeping', rotation=90, verticalalignment='center')
axes_corr_all_sj_groupby_window.axvline(x = 22.5, color='r', linestyle='--')
axes_corr_all_sj_groupby_window.text(21.5, 140, 'Sleeping_to_Activity', rotation=90, verticalalignment='center')
axes_corr_all_sj_groupby_window.errorbar(corr_all_sj_groupby_window.index, corr_all_sj_groupby_window['mean_HR_empatica'], yerr=corr_all_sj_groupby_window['mae_HR_empatica'], fmt = 'x')
axes_corr_all_sj_groupby_window.set(ylabel='Empatica - Heart rate(bmp)', xlabel='windows')

fig.savefig(path_img + "all_devices_states_hr_with_error_to_standard")
"""
# By devices & windows (1 subject)
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
axes_stderr_resting.errorbar(corr_resting_df.index, corr_resting_df['mean_HR_fitbit'], yerr=corr_resting_df['mae_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_resting.set(ylabel='Fitbit - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_resting = plt.subplot(2, 2, 2)
axes_stderr_resting.errorbar(corr_resting_df['n_window'], corr_resting_df['mean_HR_empatica'], yerr=corr_resting_df['mae_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_resting.set(ylabel='Empatica - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_resting = plt.subplot(2, 2, 3)
axes_stderr_resting.errorbar(corr_resting_df['n_window'], corr_resting_df['mean_HR_fitbit'], yerr=corr_resting_df['mae_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_resting.errorbar(corr_resting_df['n_window'], corr_resting_df['mean_HR_empatica'], yerr=corr_resting_df['mae_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_resting.set(ylabel='Heart rate(bpm)', xlabel='n_windows')
plt.legend()
fig.savefig(path_img + subject_folder + '_stderr_resting_state', quality=95)

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
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_fitbit'], yerr=corr_sleeping_df['mae_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.set(ylabel='Fitbit - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_sleeping = plt.subplot(2, 2, 2)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_empatica'], yerr=corr_sleeping_df['mae_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.set(ylabel='Empatica - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_sleeping = plt.subplot(2, 2, 3)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_emfitqs'], yerr=corr_sleeping_df['mae_HR_emfitqs'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.set(ylabel='EmfitQS - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_sleeping = plt.subplot(2, 2, 4)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_fitbit'], yerr=corr_sleeping_df['mae_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_empatica'], yerr=corr_sleeping_df['mae_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_sleeping.errorbar(corr_sleeping_df['n_window'], corr_sleeping_df['mean_HR_emfitqs'], yerr=corr_sleeping_df['mae_HR_emfitqs'], fmt='x', elinewidth=2.5, markersize=10)
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
axes_stderr_activity.errorbar(corr_activity_df['n_window'], corr_activity_df['mean_HR_fitbit'], yerr=corr_activity_df['mae_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_activity.set(ylabel='Fitbit - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_activity = plt.subplot(2, 2, 2)
axes_stderr_activity.errorbar(corr_activity_df['n_window'], corr_activity_df['mean_HR_empatica'], yerr=corr_activity_df['mae_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_activity.set(ylabel='Empatica - Heart rate(bpm)', xlabel='n_windows')

axes_stderr_activity = plt.subplot(2, 2, 3)
axes_stderr_activity.errorbar(corr_activity_df['n_window'], corr_activity_df['mean_HR_fitbit'], yerr=corr_activity_df['mae_HR_fitbit'], fmt='o', elinewidth=2.5, markersize=10)
axes_stderr_activity.errorbar(corr_activity_df['n_window'], corr_activity_df['mean_HR_empatica'], yerr=corr_activity_df['mae_HR_empatica'], fmt='x', elinewidth=2.5, markersize=10)
axes_stderr_activity.set(ylabel='Heart rate(bpm)', xlabel='n_windows')
plt.legend()
fig.savefig(path_img + subject_folder + '_stderr_activity_state', quality=95)
"""

# 3. Correlation with accelerometer 
fig = plt.figure()
my_dpi = 96
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi))
plt.subplots_adjust(top=0.905,
bottom=0.06,
left=0.08,
right=0.955,
hspace=0.225,
wspace=0.155)
fig.suptitle(subject_folder + ' - Devices Correlation Coefficient with Accelerometer', fontsize=40)
axes_devices_corr = plt.subplot(2, 2, 1)
axes_devices_corr.errorbar(['Resting', 'Sleeping', 'Activity'], [corr_resting_df['corr_VectorA_HR_fitbit'].mean(), corr_sleeping_df['corr_VectorA_HR_fitbit'].mean(), corr_activity_df['corr_VectorA_HR_fitbit'].mean()], 
              yerr=[np.std(corr_resting_df['corr_VectorA_HR_fitbit']), np.std(corr_sleeping_df['corr_VectorA_HR_fitbit']), np.std(corr_activity_df['corr_VectorA_HR_fitbit'])], 
              fmt='x', elinewidth=2.5, markersize=10)
axes_devices_corr.set(ylabel='Fitbit Correlation Coefficient with Accelerometer', xlabel='States')

axes_devices_corr = plt.subplot(2, 2, 2)
axes_devices_corr.errorbar(['Resting', 'Sleeping', 'Activity'], [corr_resting_df['corr_VectorA_HR_empatica'].mean(), corr_sleeping_df['corr_VectorA_HR_empatica'].mean(), corr_activity_df['corr_VectorA_HR_empatica'].mean()], 
              yerr=[np.std(corr_resting_df['corr_HR_empatica']), np.std(corr_sleeping_df['corr_VectorA_HR_empatica']), np.std(corr_activity_df['corr_VectorA_HR_empatica'])], 
              fmt='o', elinewidth=2.5, markersize=10)
axes_devices_corr.set(ylabel='Empatica Correlation Coefficient with Accelerometer', xlabel='States')

axes_devices_corr = plt.subplot(2, 2, 3)
axes_devices_corr.plot(corr_sleeping_df['n_window'], corr_sleeping_df['corr_VectorA_HR_emfitqs'], '*')
axes_devices_corr.set(ylabel='EmfitQS - Correlation Coefficient with Accelerometer', xlabel='n_windows')

axes_devices_corr = plt.subplot(2, 2, 4)
axes_devices_corr.errorbar(['Resting', 'Sleeping', 'Activity'], [corr_resting_df['corr_VectorA_HR_fitbit'].mean(), corr_sleeping_df['corr_VectorA_HR_fitbit'].mean(), corr_activity_df['corr_VectorA_HR_fitbit'].mean()], 
              yerr=[np.std(corr_resting_df['corr_VectorA_HR_fitbit']), np.std(corr_sleeping_df['corr_VectorA_HR_fitbit']), np.std(corr_activity_df['corr_VectorA_HR_fitbit'])], 
              fmt='x', elinewidth=2.5, markersize=10)
axes_devices_corr.errorbar(['Resting', 'Sleeping', 'Activity'], [corr_resting_df['corr_VectorA_HR_empatica'].mean(), corr_sleeping_df['corr_VectorA_HR_empatica'].mean(), corr_activity_df['corr_VectorA_HR_empatica'].mean()], 
              yerr=[np.std(corr_resting_df['corr_HR_empatica']), np.std(corr_sleeping_df['corr_VectorA_HR_empatica']), np.std(corr_activity_df['corr_VectorA_HR_empatica'])], 
              fmt='o', elinewidth=2.5, markersize=10)
axes_devices_corr.set(ylabel='All Devices Correlation Coefficient with Accelerometer', xlabel='States')
plt.legend()
fig.savefig(path_img + "all_devices_states_error_hr_to_accelerometer")

# Exporting Correlation Matrix
export_columns = ['corr_HR_empatica', 'p_score_HR_empatica',
       't_stat_HR_empatica', 'std_HR_empatica', 'mean_HR_empatica',
       'mse_HR_empatica', 'rmse_HR_empatica', 'sem_HR_empatica',
       'corr_VectorA_HR_empatica', 'mae_HR_empatica', 
       'corr_HR_fitbit', 'p_score_HR_fitbit', 't_stat_HR_fitbit',
       'std_HR_fitbit', 'mean_HR_fitbit', 'mse_HR_fitbit', 'rmse_HR_fitbit',
       'sem_HR_fitbit', 'corr_VectorA_HR_fitbit', 'mae_HR_fitbit']
corr_resting_groupby_window[export_columns].describe().to_excel(path_img  + 'all_sj_resting_summary.xlsx')
corr_sleeping_groupby_window[export_columns].describe().to_excel(path_img  + 'all_sj_sleeping_summary.xlsx')
corr_activity_groupby_window[export_columns].describe().to_excel(path_img  + 'all_sj_activity_summary.xlsx')
corr_all_sj_groupby_window[export_columns].describe().to_excel(path_img + 'all_sj_group_window.xlsx')
