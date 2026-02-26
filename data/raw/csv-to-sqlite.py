import pandas as pd
import sqlite3
import os

# Chemins des fichiers (ajuste selon ta structure)
data_dir = "."
db_path = "mobility.db"

def build_db():
    conn = sqlite3.connect(db_path)
    
    # 1. Charger les collisions
    if os.path.exists(f"{data_dir}/collisions.csv"):
        df_coll = pd.read_csv(f"{data_dir}/collisions.csv")
        df_coll.to_sql("collisions", conn, if_exists="replace", index=False)
        print("[OK] Table 'collisions' créée.")

    # 2. Charger le 311
    if os.path.exists(f"{data_dir}/requetes_311.csv"):
        df_311 = pd.read_csv(f"{data_dir}/requetes_311.csv", low_memory=False)
        df_311.to_sql("requetes_311", conn, if_exists="replace", index=False)
        print("[OK] Table 'requetes_311' créée.")
    
    if os.path.exists(f"{data_dir}/weather_montreal.csv"):
        df_311 = pd.read_csv(f"{data_dir}/weather_montreal.csv", low_memory=False)
        df_311.to_sql("weather_montreal", conn, if_exists="replace", index=False)
        print("[OK] Table 'weather_montreal' créée.")

    conn.close()

if __name__ == "__main__":
    build_db()
