"""
Module pour générer les briefings automatiques.
Combine hotspots, tendances, signaux et recommandations.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

from .hotspots import get_top_hotspots_collisions, get_top_hotspots_311, get_hotspot_recommendations
from .trends import analyze_trends_yoy, analyze_trends_mom
from .weak_signals import detect_weak_signals, detect_anomalies_zscore
from .formatter import format_briefing_public, format_briefing_municipality

logger = logging.getLogger(__name__)


def generate_briefing(df_coll: pd.DataFrame, df_311: pd.DataFrame, 
                     df_meteo: pd.DataFrame = None,
                     briefing_type: str = 'weekly',
                     target_audience: str = 'public') -> str:
    """
    Génère un briefing complet automatiquement.
    
    Args:
        df_coll: DataFrame collisions
        df_311: DataFrame 311
        df_meteo: DataFrame météo (optionnel)
        briefing_type: 'daily', 'weekly', 'monthly'
        target_audience: 'public' ou 'municipality'
    
    Returns:
        Markdown formaté du briefing
    """
    
    logger.info(f"📋 Génération du briefing {briefing_type} ({target_audience})...")
    
    # Déterminer la plage de dates selon le type
    today = datetime.now()
    if briefing_type == 'daily':
        date_start = (today - timedelta(days=1)).strftime('%Y-%m-%d')
        date_end = today.strftime('%Y-%m-%d')
        period_label = "Dernier jour"
    elif briefing_type == 'weekly':
        date_start = (today - timedelta(days=7)).strftime('%Y-%m-%d')
        date_end = today.strftime('%Y-%m-%d')
        period_label = "Dernière semaine"
    elif briefing_type == 'monthly':
        date_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
        date_end = today.strftime('%Y-%m-%d')
        period_label = "Mois dernier"
    else:
        date_start = None
        date_end = None
        period_label = "Toutes données"
    
    # Collecter tous les éléments du briefing
    briefing_data = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period_label,
        'date_start': date_start,
        'date_end': date_end,
        'briefing_type': briefing_type,
    }
    
    # 1. Top 5 Hotspots Collisions
    logger.info("  → Calcul des hotspots collisions...")
    hotspots_coll = get_top_hotspots_collisions(df_coll, n_clusters=5, 
                                                date_start=date_start, date_end=date_end)
    briefing_data['hotspots_collisions'] = hotspots_coll
    
    # 2. Top 5 Hotspots 311
    logger.info("  → Calcul des hotspots 311...")
    hotspots_311 = get_top_hotspots_311(df_311, n_hotspots=5,
                                        date_start=date_start, date_end=date_end)
    briefing_data['hotspots_311'] = hotspots_311
    
    # 3. Tendances
    logger.info("  → Analyse des tendances...")
    trends_yoy = analyze_trends_yoy(df_coll, 'DATE')
    trends_mom = analyze_trends_mom(df_coll, 'DATE', last_n_months=6)
    briefing_data['trends_yoy'] = trends_yoy
    briefing_data['trends_mom'] = trends_mom
    
    # 4. Signaux faibles
    logger.info("  → Détection des signaux faibles...")
    weak_signals = detect_weak_signals(df_311, 'DATE', 
                                       'ACTI_NOM', window_weeks=6)
    briefing_data['weak_signals'] = weak_signals[:3]  # Top 3
    
    # 5. Anomalies récentes
    logger.info("  → Détection des anomalies...")
    if briefing_type == 'weekly':
        df_coll_recent = df_coll.copy()
        df_coll_recent['DATE'] = pd.to_datetime(df_coll_recent['DATE'], errors='coerce')
        df_daily = df_coll_recent.groupby(df_coll_recent['DATE'].dt.date).size()
        anomalies = detect_anomalies_zscore(df_daily.reset_index(name='count'), 
                                           'DATE', 'count', threshold=2.5)
        briefing_data['anomalies'] = anomalies[:3]
    else:
        briefing_data['anomalies'] = []
    
    # 6. Recommandations
    logger.info("  → Génération des recommandations...")
    recommendations = get_hotspot_recommendations(hotspots_coll, hotspots_311)
    briefing_data['recommendations'] = recommendations
    
    # 7. Données globales
    briefing_data['total_collisions'] = len(df_coll)
    briefing_data['total_311'] = len(df_311)
    
    logger.info(f"✅ Briefing généré avec succès")
    
    # Formater selon l'audience
    if target_audience == 'public':
        return format_briefing_public(briefing_data)
    elif target_audience == 'municipality':
        return format_briefing_municipality(briefing_data)
    else:
        return format_briefing_public(briefing_data)


def generate_summary_stats(briefing_data: Dict) -> Dict:
    """
    Génère des statistiques de résumé pour le briefing.
    
    Returns:
        Dict avec KPIs principaux
    """
    
    stats = {
        'total_collisions': briefing_data.get('total_collisions', 0),
        'total_311': briefing_data.get('total_311', 0),
        'top_hotspot_collisions': briefing_data['hotspots_collisions'][0] if briefing_data['hotspots_collisions'] else None,
        'top_hotspot_311': briefing_data['hotspots_311'][0] if briefing_data['hotspots_311'] else None,
        'weak_signals_count': len(briefing_data.get('weak_signals', [])),
        'anomalies_count': len(briefing_data.get('anomalies', [])),
    }
    
    return stats
