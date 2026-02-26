#!/usr/bin/env python3
"""Script de test du briefing"""

import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from src.reports import generate_briefing

# Charger toutes les données
engine = create_engine('sqlite:///./data/raw/mobility.db')
df_coll = pd.read_sql_query('SELECT * FROM collisions', engine)
df_311 = pd.read_sql_query('SELECT * FROM requetes_311', engine)

print(f'Total collisions: {len(df_coll)}')
print(f'Total 311: {len(df_311)}')

# Générer le briefing
try:
    print("\n=== Génération du briefing ===")
    briefing = generate_briefing(df_coll, df_311, briefing_type='weekly', target_audience='public')
    
    print("\n=== BRIEFING OUTPUT ===")
    lines = briefing.split('\n')
    for i, line in enumerate(lines):
        print(f'{i:3d}: {line}')
        
except Exception as e:
    print(f'ERREUR: {e}')
    import traceback
    traceback.print_exc()
