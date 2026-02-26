# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 01:52:42 2026

@author: amigo
"""

import pandas as pd
def clean_coll():
    df_coll = pd.read_csv("../../data/raw/collisions_clean.csv")

    df_coll['DT_ACCDN'] = pd.to_datetime(df_coll['DT_ACCDN'])
    
    df_coll.to_csv("../../data/raw/collisions_clean.csv")

def clean_weather():
    df_meteo = pd.read_csv("../../data/raw/weather_montreal.csv")

    df_coll['DT_ACCDN'] = pd.to_datetime(df_coll['DT_ACCDN'])
    
    df_coll.to_csv("../../data/raw/collisions_clean.csv")
    
def rename_col(df, old , new):
    df.rename(columns ={old:new} , inplace=True)
    
"""
df_coll = pd.read_csv("../../data/raw/collisions_clean.csv")

rename_col(df_coll, 'DT_ACCDN' , 'DATE')

print(df_coll.columns)
df_coll.to_csv("../../data/raw/collisions_clean.csv")

df_coll = pd.read_csv("../../data/raw/stm/stops.txt")

rename_col(df_coll, 'stop_lat' , 'LOC_LAT')
rename_col(df_coll, 'stop_lon' , 'LOC_LONG')

print(df_coll.columns)
df_coll.to_csv("../../data/raw/stm.csv")"""


df_coll = pd.read_csv("../../data/raw/requetes_311.csv")

rename_col(df_coll, 'DDS_DATE_CREATION' , 'DATE')

print(df_coll.columns)
df_coll.to_csv("../../data/raw/requetes_311.csv")