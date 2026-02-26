"""
Module pour identifier les top 5 hotspots (zones concentrées d'incidents).
Utilise clustering spatial et agrégations Pandas.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def get_top_hotspots_collisions(df_coll: pd.DataFrame, n_clusters: int = 5, 
                                date_start: str = None, date_end: str = None) -> List[Dict]:
    """
    Identifie les top 5 hotspots de collisions par clustering spatial K-Means.
    
    Args:
        df_coll: DataFrame collisions avec LOC_LAT, LOC_LONG
        n_clusters: Nombre de zones à identifier
        date_start, date_end: Filtres de dates optionnels (format 'YYYY-MM-DD')
    
    Returns:
        Liste de dicts : {zone_id, center_lat, center_lon, count, graves, morts, coords}
    """
    
    logger.info(f"🔍 Identification des {n_clusters} hotspots collisions...")
    
    # Copie et nettoyage
    df = df_coll.copy()
    df = df.dropna(subset=['LOC_LAT', 'LOC_LONG'])
    
    # Filtrage temporel si spécifié
    if date_start and date_end:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df_filtered = df[(df['DATE'] >= date_start) & (df['DATE'] <= date_end)]
        # Si aucune donnée dans la période, utiliser toutes les données
        if len(df_filtered) > 0:
            df = df_filtered
        else:
            logger.warning(f"⚠️  Aucune donnée collisions entre {date_start} et {date_end}, utilisant toutes les données")
    
    if len(df) == 0:
        logger.warning("⚠️  Aucune donnée collisions à analyser")
        return []
    
    # Clustering spatial K-Means
    coords = df[['LOC_LAT', 'LOC_LONG']].values
    kmeans = KMeans(n_clusters=min(n_clusters, len(df)), random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(coords)
    
    # Agrégation par cluster
    hotspots = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            continue
        
        # Conversion sécurisée des colonnes numériques
        nb_morts = pd.to_numeric(cluster_data['NB_MORTS'], errors='coerce').fillna(0).sum()
        nb_blesses_graves = pd.to_numeric(cluster_data['NB_BLESSES_GRAVES'], errors='coerce').fillna(0).sum()
        
        n_graves = (cluster_data['GRAVITE'] == 'Grave').sum() if 'GRAVITE' in cluster_data.columns else 0
        
        hotspots.append({
            'zone_id': cluster_id,
            'center_lat': float(kmeans.cluster_centers_[cluster_id][0]),
            'center_lon': float(kmeans.cluster_centers_[cluster_id][1]),
            'count': len(cluster_data),
            'graves': int(n_graves),
            'morts': int(nb_morts),
            'blesses_graves': int(nb_blesses_graves),
            'pct_graves': round(100 * n_graves / len(cluster_data), 1),
            'observations': cluster_data[['DATE', 'LOC_LAT', 'LOC_LONG', 'GRAVITE']].to_dict('records')[:3]
        })
    
    # Trier par nombre total d'incidents
    hotspots = sorted(hotspots, key=lambda x: x['count'], reverse=True)[:n_clusters]
    
    logger.info(f"✅ {len(hotspots)} hotspots identifiés")
    return hotspots


def get_top_hotspots_311(df_311: pd.DataFrame, motif_col: str = 'ACTI_NOM', 
                         n_hotspots: int = 5, date_start: str = None, 
                         date_end: str = None) -> List[Dict]:
    """
    Identifie les top 5 zones avec le plus de requêtes 311 par type.
    
    Args:
        df_311: DataFrame 311
        motif_col: Colonne de catégories ('ACTI_NOM' ou 'NATURE')
        n_hotspots: Nombre de zones à retourner
        date_start, date_end: Filtres de dates (format 'YYYY-MM-DD')
    
    Returns:
        Liste de dicts : {motif, arrondissement, count, priority}
    """
    
    logger.info(f"🔍 Identification des {n_hotspots} hotspots 311 ({motif_col})...")
    
    df = df_311.copy()
    
    # Filtrage temporel
    if date_start and date_end:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df_filtered = df[(df['DATE'] >= date_start) & (df['DATE'] <= date_end)]
        # Si aucune donnée dans la période, utiliser toutes les données
        if len(df_filtered) > 0:
            df = df_filtered
        else:
            logger.warning(f"⚠️  Aucune donnée 311 entre {date_start} et {date_end}, utilisant toutes les données")
    
    if len(df) == 0:
        logger.warning("⚠️  Aucune donnée 311 à analyser")
        return []
    
    # Agrégation par motif + arrondissement
    df[motif_col] = df[motif_col].fillna('Inconnu')
    hotspots_agg = df.groupby([motif_col, 'ARRONDISSEMENT']).size().reset_index(name='count')
    hotspots_agg = hotspots_agg.sort_values('count', ascending=False).head(n_hotspots)
    
    hotspots = []
    max_count = hotspots_agg['count'].max()
    
    for _, row in hotspots_agg.iterrows():
        # Calculer priorité (urgencem basée sur volume et tendance)
        priority = 'Haute' if row['count'] > max_count * 0.6 else 'Moyenne' if row['count'] > max_count * 0.2 else 'Basse'
        
        hotspots.append({
            'motif': str(row[motif_col]),
            'arrondissement': str(row['ARRONDISSEMENT']),
            'count': int(row['count']),
            'priority': priority,
            'pct_of_total': round(100 * row['count'] / df.shape[0], 1)
        })
    
    logger.info(f"✅ {len(hotspots)} hotspots 311 identifiés")
    return hotspots


def get_hotspot_recommendations(hotspots_collisions: List[Dict], 
                                hotspots_311: List[Dict]) -> List[str]:
    """
    Génère des recommandations basées sur les hotspots identifiés.
    
    Returns:
        Liste de recommandations textuelles
    """
    
    recommendations = []
    
    # Recommandations pour les hotspots collisions
    for i, spot in enumerate(hotspots_collisions[:3], 1):
        if spot['graves'] > 0:
            recommendations.append(
                f"🚨 Zone #{i} (Lat: {spot['center_lat']:.2f}, Lon: {spot['center_lon']:.2f}): "
                f"{spot['count']} accidents ({spot['graves']} graves) → "
                f"Renforcer signalisation et marquage au sol"
            )
    
    # Recommandations pour 311
    for i, spot in enumerate(hotspots_311[:2], 1):
        if spot['priority'] == 'Haute':
            recommendations.append(
                f"🔧 {spot['arrondissement']}: {spot['count']} requêtes '{spot['motif']}' → "
                f"Augmenter fréquence d'intervention"
            )
    
    return recommendations
