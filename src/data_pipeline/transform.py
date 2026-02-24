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

def clean_coll():
    df_meteo = pd.read_csv("../../data/raw/weather_montreal.csv")

    df_coll['DT_ACCDN'] = pd.to_datetime(df_coll['DT_ACCDN'])
    
    df_coll.to_csv("../../data/raw/collisions_clean.csv")