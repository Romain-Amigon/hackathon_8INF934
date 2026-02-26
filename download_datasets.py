#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de téléchargement et génération automatique des datasets
Télécharge les fichiers listés dans filetodowload et les stocke dans data/raw/
Génère également les données météo et STM
"""

import os
import sys
import pandas as pd
import requests
import zipfile
import io
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timedelta

# Configuration
FILETODOWLOAD = "filetodowload"
OUTPUT_DIR = "data/raw"
CHUNK_SIZE = 8192

def parse_filetodowload():
    """Parse le fichier filetodowload et retourne les URLs avec leurs noms"""
    urls_dict = {}
    
    if not os.path.exists(FILETODOWLOAD):
        print(f"⚠️  Fichier {FILETODOWLOAD} non trouvé, skipping...")
        return urls_dict
    
    with open(FILETODOWLOAD, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    current_key = None
    
    for line in lines:
        line = line.strip()
        
        if line.endswith(':'):
            current_key = line[:-1]
            urls_dict[current_key] = []
        elif line.startswith('https://'):
            if current_key:
                urls_dict[current_key].append(line)
    
    return urls_dict

def get_filename_from_url(url):
    """Extrait le nom du fichier à partir de l'URL"""
    parsed = urlparse(url)
    filename = parsed.path.split('/')[-1]
    return filename

def download_file(url, output_path):
    """Télécharge un fichier depuis une URL"""
    filename = output_path.split('/')[-1]
    print(f"Téléchargement: {filename[:40]}...")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"  Progression: {percent:.1f}%", end='\r')
        
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        return True
    
    except Exception as e:
        print(f"  ✗ Erreur: {str(e)}")
        return False

def fetch_stm_data():
    """Télécharge les données GTFS de la STM"""
    print("\n📍 Téléchargement données STM (GTFS)...")
    url = "https://www.stm.info/sites/default/files/gtfs/gtfs_stm.zip"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        stm_dir = os.path.join(OUTPUT_DIR, "stm")
        os.makedirs(stm_dir, exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extract("stops.txt", path=stm_dir)
            z.extract("routes.txt", path=stm_dir)
        
        print(f"  ✓ STM data sauvegardé dans {stm_dir}")
        return True
    except Exception as e:
        print(f"  ✗ Erreur STM: {str(e)}")
        return False

def fetch_weather_data():
    """Télécharge les données météorologiques depuis Open-Meteo"""
    print("\n🌡️  Téléchargement données météo (Open-Meteo)...")
    
    try:
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
        
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        df_weather = pd.DataFrame(data['daily'])
        
        output_path = os.path.join(OUTPUT_DIR, "weather_montreal.csv")
        df_weather.to_csv(output_path, index=False)
        
        print(f"  ✓ weather_montreal.csv ({len(df_weather)} lignes)")
        return True
    except Exception as e:
        print(f"  ✗ Erreur météo: {str(e)}")
        return False

def fuse_311_files():
    """Fusionne les fichiers 311 téléchargés"""
    print("\n🔗 Fusion des fichiers 311...")
    
    categories = [
        'Feux de circulation - Entretien', 'Signalisation - Circulation - Études',
        'Collecte de déchets', 'Bac roulant', "Fermeture d'une conduite d'eau - Urgence",
        'Collecte des encombrants', 'Info-Remorquage', 'Neige - Remorquage',
        'Opération déneigement', 'Nid-de-poule', "Fermeture d'entrée d'eau",
        'Permis - Divers', 'Info-travaux - Service central', 'Collecte des matières recyclables', 
        'Branche tombée', 'Danger potentiel - Arbre', "Info - Collecte d'objets encombrants",
        'Collecte des matières organiques', 'Permis - Neige - Domaine public',
        'Info-travaux - Arrondissement', 'Intervention stationnement', 'Stationnement municipal',
        'Carte Info-Neige', 'Trottoir glissant', 'Chaussée glissante',
        'Piste cyclable déneigement', 'Pavage - Réparation', 'Trottoir ou bordure - Réparation',
        "Fuite d'eau", 'Égout - Crue des eaux', 'Signalisation endommagée Vm',
        '*Signalisation manquante ou endommagée', '*Circulation - Travaux majeurs',
        '*Entrave - Piste cyclable', 'Débris sur la voie publique', 'Arbre tombé'
    ]
    
    df_clean_list = []
    fichiers_311 = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('requetes311')]
    
    if not fichiers_311:
        print("  ⚠️  Aucun fichier requetes311 trouvé")
        return False
    
    for fichier in fichiers_311:
        filepath = os.path.join(OUTPUT_DIR, fichier)
        try:
            df = pd.read_csv(filepath, low_memory=False)
            
            if 'ACTI_NOM' in df.columns:
                df_filtered = df[df['ACTI_NOM'].isin(categories)].copy()
                
                if 'LOC_LAT' in df_filtered.columns and 'LOC_LONG' in df_filtered.columns:
                    df_filtered = df_filtered.dropna(subset=['LOC_LAT', 'LOC_LONG'])
                
                df_clean_list.append(df_filtered)
                print(f"  + {fichier} ({len(df_filtered)} lignes)")
        except Exception as e:
            print(f"  ✗ Erreur lecture {fichier}: {str(e)}")
    
    if not df_clean_list:
        print("  ⚠️  Aucun données valides trouvées")
        return False
    
    df_final = pd.concat(df_clean_list, ignore_index=True)
    df_final = df_final.drop_duplicates()
    
    if 'DATE_DERNIER_STATUT' in df_final.columns:
        df_final['DATE_DERNIER_STATUT'] = pd.to_datetime(df_final['DATE_DERNIER_STATUT'], errors='coerce')
    
    output_path = os.path.join(OUTPUT_DIR, "requetes_311_all_time.csv")
    df_final.to_csv(output_path, index=False)
    
    print(f"  ✓ requetes_311_all_time.csv ({len(df_final)} lignes)")
    return True

def create_collision_alias():
    """Crée un alias collisions.csv pour collisions_routieres.csv"""
    print("\n📋 Création des fichiers CSV requis...")
    
    collision_file = os.path.join(OUTPUT_DIR, "collisions.csv")
    collision_routieres = os.path.join(OUTPUT_DIR, "collisions_routieres.csv")
    
    # Si collisions.csv n'existe pas mais collisions_routieres.csv existe, créer l'alias
    if not os.path.exists(collision_file) and os.path.exists(collision_routieres):
        try:
            import shutil
            shutil.copy(collision_routieres, collision_file)
            print(f"  ✓ Alias créé: collisions.csv")
            return True
        except Exception as e:
            print(f"  ✗ Erreur création alias: {str(e)}")
            return False
    elif os.path.exists(collision_file):
        print(f"  ✓ collisions.csv existe déjà")
        return True
    return False

def build_sqlite_db():
    """Crée la base de données SQLite à partir des CSV"""
    print("\n🗄️  Construction de la base de données SQLite...")
    
    try:
        import sqlite3
        
        db_path = os.path.join(OUTPUT_DIR, "mobility.db")
        
        # Supprimer l'ancienne DB pour éviter les problèmes de permissions
        if os.path.exists(db_path):
            os.remove(db_path)
        
        conn = sqlite3.connect(db_path)
        
        tables_created = 0
        
        # 1. Table collisions (cherche les deux noms possibles)
        collision_path = os.path.join(OUTPUT_DIR, "collisions_routieres.csv")
        if not os.path.exists(collision_path):
            collision_path = os.path.join(OUTPUT_DIR, "collisions.csv")
        
        if os.path.exists(collision_path):
            try:
                df_coll = pd.read_csv(collision_path)
                df_coll.to_sql("collisions", conn, if_exists="replace", index=False)
                print(f"  ✓ Table 'collisions' créée ({len(df_coll)} lignes)")
                tables_created += 1
            except Exception as e:
                print(f"  ✗ Erreur collisions: {str(e)}")
        
        # 2. Table requetes_311
        requetes_path = os.path.join(OUTPUT_DIR, "requetes_311_all_time.csv")
        if os.path.exists(requetes_path):
            try:
                df_311 = pd.read_csv(requetes_path, low_memory=False)
                df_311.to_sql("requetes_311", conn, if_exists="replace", index=False)
                print(f"  ✓ Table 'requetes_311' créée ({len(df_311)} lignes)")
                tables_created += 1
            except Exception as e:
                print(f"  ✗ Erreur requetes_311: {str(e)}")
        
        # 3. Table weather_montreal
        weather_path = os.path.join(OUTPUT_DIR, "weather_montreal.csv")
        if os.path.exists(weather_path):
            try:
                df_weather = pd.read_csv(weather_path, low_memory=False)
                df_weather.to_sql("weather_montreal", conn, if_exists="replace", index=False)
                print(f"  ✓ Table 'weather_montreal' créée ({len(df_weather)} lignes)")
                tables_created += 1
            except Exception as e:
                print(f"  ✗ Erreur weather_montreal: {str(e)}")
        
        conn.close()
        
        if tables_created > 0:
            print(f"  ✓ Base de données SQLite créée: {db_path}")
            return True
        else:
            print(f"  ⚠️  Aucune table créée (fichiers CSV manquants)")
            return False
    
    except Exception as e:
        print(f"  ✗ Erreur base de données: {str(e)}")
        return False

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📥 TÉLÉCHARGEMENT ET GÉNÉRATION DES DATASETS - MOBILITY COPILOT")
    print("=" * 70)
    
    success_count = 0
    
    # 1. Télécharger les fichiers du filetodowload
    urls_dict = parse_filetodowload()
    if urls_dict:
        total_files = sum(len(urls) for urls in urls_dict.values())
        print(f"\n🔍 Fichiers trouvés: {total_files}")
        print("-" * 70)
        
        for category, urls in urls_dict.items():
            print(f"\n📂 {category} ({len(urls)} fichier(s))")
            for url in urls:
                filename = get_filename_from_url(url)
                output_path = os.path.join(OUTPUT_DIR, filename)
                
                if os.path.exists(output_path):
                    print(f"  ⏭️  {filename} (déjà présent)")
                    success_count += 1
                else:
                    if download_file(url, output_path):
                        success_count += 1
    
    # 2. Télécharger les données STM
    if fetch_stm_data():
        success_count += 1
    
    # 3. Télécharger les données météo
    if fetch_weather_data():
        success_count += 1
    
    # 4. Fusionner les fichiers 311
    if fuse_311_files():
        success_count += 1
    
    # 5. Créer les alias de fichiers (collisions.csv)
    if create_collision_alias():
        success_count += 1
    
    # 6. Créer la base de données SQLite
    if build_sqlite_db():
        success_count += 1
    
    # Résumé final
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLET TERMINÉ")
    print("=" * 70)
    
    print(f"\n📋 Fichiers générés dans {OUTPUT_DIR}:")
    total_size = 0
    for item in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, item)
        if os.path.isfile(path):
            size = os.path.getsize(path) / 1024 / 1024
            total_size += size
            print(f"   • {item} ({size:.1f} MB)")
        elif os.path.isdir(path):
            print(f"   📁 {item}/")
    
    print(f"\n💾 Taille totale: {total_size:.1f} MB")
    print("\n✨ Application ready! Vous pouvez maintenant lancer:")
    print("   streamlit run src/ui/app.py")

if __name__ == "__main__":
    main()