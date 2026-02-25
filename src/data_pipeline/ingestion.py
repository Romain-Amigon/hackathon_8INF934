# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 04:52:32 2026

@author: amigo
"""

import os
import pandas as pd
import requests
import zipfile
import io
from datetime import datetime, timedelta
def fetch_stm_data():
    url = "https://www.stm.info/sites/default/files/gtfs/gtfs_stm.zip"
    response = requests.get(url)
    
    output_dir = os.path.join(os.path.dirname(__file__), '../../data/raw/stm')
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extract("stops.txt", path=output_dir)
        z.extract("routes.txt", path=output_dir)
    
    df_stops = pd.read_csv(os.path.join(output_dir, "stops.txt"))
    return df_stops

def fetch_collisions_data():
    url = "https://donnees.montreal.ca/dataset/cd722e22-376b-4b89-9bc2-7c7ab317ef6b/resource/05deae93-d9fc-4acb-9779-e0942b5e962f/download/collisions_routieres.csv"
    df = pd.read_csv(url)
    
    colonnes_utiles = [
        'DT_ACCDN', 'HR_ACCDN', 'HEURE_ACCDN', 'GRAVITE', 'NB_VICTIMES_TOTAL', 'NB_MORTS', 
        'NB_BLESSES_GRAVES', 'NB_BLESSES_LEGERS', 'CD_ETAT_SURFC', 
        'CD_ECLRM', 'CD_ENVRN_ACCDN', 'LOC_LAT', 'LOC_LONG'
    ]
    
    colonnes_presentes = [col for col in colonnes_utiles if col in df.columns]
    df_clean = df[colonnes_presentes].copy()
    
    if 'LOC_LAT' in df_clean.columns and 'LOC_LONG' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['LOC_LAT', 'LOC_LONG'])
    
    output_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
    os.makedirs(output_dir, exist_ok=True)
    df_clean.to_csv(os.path.join(output_dir, "collisions.csv"), index=False)
    
    return df_clean

def fetch_weather_data():
    #date_debut =(datetime.now() - timedelta(weeks=124)).strftime('%Y-%m-%d')
    date_fin = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 45.5017,
        "longitude": -73.5673,
        "start_date": "2012-01-01",
        "end_date": date_fin,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "snowfall_sum"],
        "timezone": "America/New_York"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    df_weather = pd.DataFrame(data['daily'])
    
    output_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
    os.makedirs(output_dir, exist_ok=True)
    df_weather.to_csv(os.path.join(output_dir, "weather_montreal.csv"), index=False)
    
    return df_weather

def main():
    fetch_stm_data()
    fetch_collisions_data()
    fetch_weather_data()

main()