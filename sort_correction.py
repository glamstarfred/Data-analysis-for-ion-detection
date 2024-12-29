#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:20:11 2023

@author: haiyue
"""

import pandas as pd
import datetime
import os   
import numpy as np
import aacgmv2
import glob
import time
import tkinter                 
from tkinter import filedialog as fd
#outpath = "../Level1/"     #specify the save path
#if not os.path.exists(outpath):
    #os.mkdir(outpath)


desired_columns = [
    2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,28,29,40,41,42,43,
    44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,69,70,71,72,73
    ]


# df = pd.read_csv("BD3G03__2021-09-24_GEO_coordinates.csv")

def level0_to_level1(file):
    df = pd.read_csv(file)
    
    df = df.iloc[:, desired_columns]


    df = df.drop(index=(df.loc[(df['raw data BCC'] != df['data BCC'])].index))
    df = df.drop(index=(df.loc[(df['raw setting BCC'] != df['setting BCC'])].index))

    df = df.drop_duplicates()

    df.drop(df.iloc[:, 2:18], axis=1, inplace=True)
    df.drop(df.iloc[:, 4:8],axis=1, inplace=True)

    df = df.sort_values(by=['timestamp (UTC)','Group Number','Step Number'])
    df = df.reset_index(drop = True)
    
    
    
    b = df["Group Number"]*2048 + df["Step Number"]
    a = np.diff(b)
    ori_sert_loc = np.argwhere(a > 1)[:,0]
    sert_amount = a[a > 1] - 1
    add_num = [sum(sert_amount[:i]) for i in range(len(sert_amount))]
    sert_loc = add_num + ori_sert_loc

    #print(a,ori_sert_loc,sert_amount,sert_loc)

    for i in range(0, len(sert_loc)):
        insert = pd.DataFrame(-999.9, columns = df.columns, index=[sert_loc[i]] * sert_amount[i])
        df = pd.concat([df[0:sert_loc[i]+1], insert, df[sert_loc[i]+1:len(df)]])

    df_fill = df.reset_index(drop = True) 

 #    Group Number and Step Number and Datatime

    Steploop_N = np.arange(2048)
    Step_new   = np.tile(Steploop_N,(512,1))
    Step_new = Step_new.flatten()
    
    
    Grouploop  = np.arange(256) 
    Group_new  = np.tile(Grouploop,(2048,1))
    Group_new  = Group_new.T.flatten()
    Group_new = np.append(Group_new, Group_new)
    
    
    Time_Step_add = np.arange( 2048*256*2 ) * 0.311401
    Time_Group_add_gap = np.arange(256*2)*4.75      #make a gap of +4.75s for each group to avoid error
    Time_Group_add_gap = np.tile(Time_Group_add_gap,(2048,1))
    Time_Group_add = Time_Group_add_gap.T.flatten()
    Time_add = Time_Group_add + Time_Step_add

    Start_SN = int(df_fill['Step Number'][0])
    Start_Group = int(df_fill['Group Number'][0])
    Start_Order = Start_Group*2048+ Start_SN
    Snum_replace = Step_new[Start_Order : Start_Order+ len(df_fill)]
    Group_replace =  Group_new[Start_Order : Start_Order+ len(df_fill)]
    
    Start_Time =  df_fill['timestamp (UTC)'][0]
    Time_replace  = Time_add[Start_Order : Start_Order+ len(df_fill)] 
    Time_replace = Time_replace- Time_replace[0] + Start_Time
    
    
    timeStamp = Time_replace
    dateArray = pd.to_datetime(timeStamp,unit='s')
    df2 = pd.DataFrame(dateArray)
    df2.columns = ['date']
    DataTime_UTC = df2.date.dt.strftime('%Y-%m-%d/%H:%M:%S')
    
    
 #  save

    df_fill = df_fill.drop(columns = ['Step Number','Group Number', 'timestamp (UTC)','Date Time (UTC)'] )
    
    df_fill.insert(0,'Group Number', Group_replace)
    df_fill.insert(0,'Step Number', Snum_replace)
    df_fill.insert(20,'Date Time (UTC)', DataTime_UTC)
    df_fill.insert(20,'timestamp (UTC)', Time_replace)


    Mlt_arr = list(map(lambda x:aacgmv2.wrapper.convert_mlt(-177.0, x), dateArray))
    Mlt_arr = pd.DataFrame( Mlt_arr, columns =['MLT'])

    df_fill = pd.concat([df_fill, Mlt_arr], axis = 1)
    df_fill.to_csv('BD-G3-QXLZ-Level-1-'+save_date+'-IIItest.csv', index=False)
    print(datetime.datetime.now())

# In[]
if __name__ == "__main__":
    root = tkinter.Tk()
    filepath = fd.askdirectory(title='Open Level 0 files',initialdir='/') + '/'
    root.destroy() 
    
    
    files = sorted(glob.glob(filepath +'*.csv'))     
    for j in range(0,len(files)):
        save_date = files[j][len(filepath)+8 :  len(filepath)+18 ]
    
        level0_to_level1(files[j])








#df = pd.read_csv("BD3G03__2021-09-24_GEO_coordinates.csv")
#df_fill.to_csv('BD3G03__2021-09-24_GEO_coordinates_level_1.csv', index=False)
