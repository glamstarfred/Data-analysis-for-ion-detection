#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:08:32 2024

@author: haiyue
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 17:11:11 2023

@author: haiyue
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import datetime
from dateutil.parser import parse
import glob
import tkinter                 
from tkinter import filedialog as fd
#outpath = "../Level1/"     #创建结果保存目录，需要自己修改，默认位置是程序所在位置

font = {'family' : 'Arial',
        'weight' : 'light',
        'size'   : 14}
size = 16

mpl.rc('font', **font)
colunm = ['Group Number','Step Number','Ch0','Ch1','Ch2','Ch3','Ch4','Ch5','Ch6','Ch7','Ch8','Ch9','Ch10',
      'Ch11','Ch12','Ch13','Ch14','Ch15', 'Date Time (UTC)']

channels = ['Ch0','Ch1','Ch2','Ch3','Ch4','Ch5','Ch6','Ch7','Ch8','Ch9','Ch10',
      'Ch11','Ch12','Ch13','Ch14', 'Ch15']




ratio_factor = [0.8485, 0.8993, 0.9277, 0.9516, 0.9749, 0.9882, 0.9981, \
                1.0, 0.9882, 0.9665, 0.9341, 0.8915, 0.8577, 0.8295, 0.7885]
    
E_level      = [5.10, 5.84, 6.69, 7.66, 8.77, 10.04, 11.50, 13.17, 15.08, 17.27,\
                19.77, 22.64, 25.92, 29.68, 33.99, 38.92, 44.57, 51.04, 58.44,\
                66.92, 76.63, 87.74, 100.47, 115.05, 131.74, 150.85, 172.73, 197.79,\
                226.49, 259.34, 296.97, 340.05, 389.38, 445.87, 510.56, 584.62,\
                669.44, 766.56, 877.76, 1005.10, 1150.93, 1317.89, 1509.09,\
                1728.01, 1978.70, 2265.76, 2594.47, 2970.85, 3401.85, 3895.37,\
                4460.49, 5107.59, 5848.57, 6697.05, 7668.63, 8781.15, 10055.06,\
                11513.82, 13184.16, 15096.86, 17287.01, 19794.89, 22666.68, 25955.00]
    
G_factor     = 3.94E-4

def connect(filename):
    df = pd.read_csv(filename)
    df.replace(-999.9, 0, inplace=True)
    
    df_1 = df[colunm]
    df_total = df_1.copy()
    df_total['Date Time (UTC)'] = pd.to_datetime(df_1['Date Time (UTC)'], format='%Y-%m-%d/%H:%M:%S')
    #将时间格式转换为datetime，便于后续操作
    grouped_df = df_total.groupby('Group Number')
    for group_number, group_df in grouped_df:
        for channel in channels:
            if group_df[channel].values.size != 2048:
                padding = np.zeros(2048 - group_df[channel].values.size)
                group_df[channel] = np.concatenate((group_df[channel].values, padding))
    df_total.update(group_df)
    return df_total
    
def find_wnoise(df): 
    df_temp = pd.DataFrame()
    for channel in channels:
        df_filtered = df[(df['Step Number'] >= 1920) & (df['Step Number'] <= 2048)] #此为无电压扫描区

        df_temp = df_filtered.groupby('Group Number')[channel].mean()
        
        # 排除偏差很大的数据
        std_threshold = 3  # 设置偏差阈值
        mean_value = df_temp.mean()
        std_value = df_temp.std()
        df_temp = df_temp[(df_temp >= mean_value - std_threshold * std_value) & (df_temp <= mean_value + std_threshold * std_value)]
        
        channel_average = df_temp.mean()
        df[channel] = df[channel] - channel_average
    return df

def filt_time(df,start_time,end_time):
    start = pd.to_datetime(start_time).time()
    end = pd.to_datetime(end_time).time()
    filtered_df = df[(df['Date Time (UTC)'].dt.time >= start) & (df['Date Time (UTC)'].dt.time <= end)]
    return filtered_df

def plot_time(df,channel): #做总计数和时间的关系图
    time = df["Date Time (UTC)"]
    num = df[channel]
    XX = time
    YY = num
    title = channel + ' time count'
    
    mask = YY > 0
    XX = XX[mask]
    YY = YY[mask]
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    ax.set_title(title)
    ax.plot(XX, YY, linestyle='-', color='b')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Counts')
    #ax.set_ylim(-100, 110)
    fig.show()
    

def uv_noise(df):
    grouped_df = df.groupby('Group Number')
    for group_number, group_df in grouped_df:
        for channel in channels:
            
            matrix = group_df[channel].values.reshape(32, 64) #将数值排列成矩阵便于操作
            for i in range(32):
                if i % 2 == 0:
                    row_avg = np.mean(matrix[i, :25])
                    matrix[i, :] -= row_avg
                else:
                    row_avg = np.mean(matrix[i, -25:])
                    matrix[i, :] -= row_avg
            group_df[channel] = matrix.flatten()
        df.update(group_df)
    print(datetime.datetime.now())
    return df

def flux(df):
    df_sum_t = pd.DataFrame()
    df_sum2 = pd.DataFrame()

    # 将数据处理成合适的格式：
    grouped_df = df.groupby('Group Number')
    for group_number, group_df in grouped_df:
        for channel in channels:
            matrix = group_df[channel].values.reshape(32, 64)
            # 排除遮挡
            matrix = matrix[8:]
            # 反转奇数行
            matrix[1::2] = matrix[1::2, ::-1]
            # 对每列求和
            column_sums = matrix.sum(axis=0)
            # 将求和结果添加到数据框中
            df_sum2[channel] = column_sums
        df_sum2['Date Time (UTC)'] = group_df.iloc[0]['Date Time (UTC)']
        df_sum2['Group Number'] = [group_number] *64
        df_sum2['Step Number'] = range(0,64) #填上相应的时间、Group和台阶数
        
        df_sum_t = pd.concat([df_sum_t, df_sum2], ignore_index=True)
            
    df_sum_t = df_sum_t.sort_values(by=['Date Time (UTC)','Step Number'])

    df_sum_t['Total Number'] = df_sum_t[channels].sum(axis=1)
    
    #计算通量
    grouped_df = df_sum_t.groupby('Step Number')
    for step_number, group_df in grouped_df:
        #df_sum_t.loc[group_df.index, 'Total Flux'] = group_df['Total Number'] / (0.2 * G_factor * E_level[step_number]/1000.0)
        df_sum_t.loc[group_df.index, 'Energy Level'] = E_level[step_number]

    dates = df_sum_t['Date Time (UTC)']
    dates = dates.drop_duplicates()

    df_spec1 = df_sum_t
    grouped_df = df_spec1.groupby('Step Number')
    
    for step_number, group_df in grouped_df:
        df_spec1.loc[group_df.index, channels] = group_df[channels] / (0.2 * G_factor * E_level[step_number]/1000.0)
        
    

    for channel in channels:
        df_spec = df_spec1.pivot(index='Energy Level', columns='Date Time (UTC)', values=channel)
        df_spec = df_spec.sort_values(by='Energy Level', ascending=False)     
            
        df_spec = df_spec.head(42)  #取高能段的离子数据，一般是大于100eV
        
    return df_spec, dates

        
# In[]


        
        
        
        
        
# In[]
if __name__ == "__main__":
    root = tkinter.Tk()
    filepath = fd.askdirectory(title='Open Level 0 files',initialdir='/') + '/'
    root.destroy() 
    
    files = sorted(glob.glob(filepath + '*.csv')) 
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to combine all files
    combined_dates = []
    
    for j in range(0,len(files)):
        save_date = files[j][len(filepath)+19 :  len(filepath)+29 ]
    
        df_plot = connect(files[j])
        df_plot = find_wnoise(df_plot)
        df_plot = uv_noise(df_plot)
        df_plot, dates = flux(df_plot)
        
        combined_df = pd.concat([combined_df, df_plot],axis=1)  # Combine current file data with previous files
        combined_dates.append(dates)
        
    for channel in channels:    
        #作图
        fig, ax = plt.subplots(figsize=(15, 15))
        # 绘制色块图，设置vmin和vmax可以调颜色
        im = ax.imshow(combined_df.values, norm=colors.LogNorm(vmin=1e+1, vmax=combined_df.values.max()),cmap='jet')
        # 设置横纵坐标刻度
        ax.set_xticks(np.linspace(0, len(combined_df.columns)-1, 4))
        ax.set_yticks(np.linspace(0, len(combined_df.index)-1, 4))
        # 设置刻度标签
        ax.set_xticklabels([dates.iloc[int(i)].strftime('%H:%M:%S') for i in ax.get_xticks()], ha='right')
        ax.set_yticklabels([E_level[::-1][int(i)] for i in ax.get_yticks()])
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, format='%.0e', shrink=0.25)
        cbar.ax.tick_params(labelsize=12)
        # 设置图表标题和坐标轴标签
        ax.set_title('Flux ' + channel)
        ax.set_xlabel('Date Time (UTC)')
        ax.set_ylabel('Energy Level\eV')
        
        plt.savefig(save_date+'-' + channel +'-'+ 'flux.png')
        
