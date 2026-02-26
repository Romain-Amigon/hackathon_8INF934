"""
Module pour détecter les signaux faibles (petits changements précoces).
Détecte les anomalies et les trends croissantes régulières.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from scipy.stats import linregress
import logging

logger = logging.getLogger(__name__)


def detect_weak_signals(df: pd.DataFrame, date_col: str, category_col: str = None,
                       window_weeks: int = 6, significance_threshold: float = 0.05) -> List[Dict]:
    """
    Détecte les signaux faibles : croissance régulière sur 6+ semaines.
    
    Args:
        df: DataFrame
        date_col: Colonne de dates
        category_col: Colonne de catégories facultative
        window_weeks: Fenêtre glissante (semaines)
        significance_threshold: Seuil de significativité (p-value)
    
    Returns:
        Liste de signaux détectés : {category, trend_strength, weeks_data, recommendation}
    """
    
    logger.info("🔔 Détection des signaux faibles...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Créer période hebdomadaire
    df['WEEK'] = df[date_col].dt.isocalendar().week
    df['YEAR'] = df[date_col].dt.year
    df['YEAR_WEEK'] = df['YEAR'].astype(str) + '-W' + df['WEEK'].astype(str).str.zfill(2)
    
    weak_signals = []
    
    if category_col is None:
        # Analyser la tendance globale
        categories = [None]
    else:
        df[category_col] = df[category_col].fillna('Inconnu')
        categories = df[category_col].value_counts().head(10).index.tolist()
    
    for cat in categories:
        if cat is None:
            df_cat = df
            cat_name = "Global"
        else:
            df_cat = df[df[category_col] == cat]
            cat_name = str(cat)
        
        # Compter par semaine
        weekly_count = df_cat.groupby('YEAR_WEEK').size()
        
        if len(weekly_count) < window_weeks:
            continue  # Pas assez de données
        
        # Garder les dernières window_weeks
        recent_weeks = weekly_count.tail(window_weeks)
        
        # Régresssion linéaire
        x = np.arange(len(recent_weeks))
        y = recent_weeks.values
        
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
        except:
            continue
        
        # Déterminer si c'est un signal faible
        # Critères : pente positive, statistiquement significative, croissance régulière
        is_weak_signal = (
            slope > 0 and  # Croissance
            p_value < significance_threshold and  # Significatif statistiquement
            np.std(y) < np.mean(y)  # Pas trop volatil
        )
        
        if is_weak_signal:
            trend_strength = slope / np.mean(y) * 100 if np.mean(y) > 0 else 0
            
            weak_signals.append({
                'category': cat_name,
                'slope': round(slope, 2),
                'trend_strength': round(trend_strength, 1),
                'p_value': round(p_value, 4),
                'r_squared': round(r_value ** 2, 3),
                'weeks_data': recent_weeks.to_dict(),
                'recommendation': f"⚠️ {cat_name}: Hausse régulière de {trend_strength:.0f}% sur {window_weeks} semaines. Surveillance recommandée."
            })
    
    logger.info(f"✅ {len(weak_signals)} signaux faibles détectés")
    return weak_signals


def detect_anomalies_zscore(df: pd.DataFrame, date_col: str, value_col: str,
                           threshold: float = 2.5) -> List[Dict]:
    """
    Détecte les anomalies (valeurs aberrantes) par Z-score.
    
    Args:
        df: DataFrame
        date_col: Colonne de dates
        value_col: Colonne à analyser (ex: nombre d'accidents par jour)
        threshold: Seuil Z-score (2.5 = très probable anomalie)
    
    Returns:
        Liste d'anomalies : {date, value, zscore}
    """
    
    logger.info("🚨 Détection des anomalies...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, value_col])
    
    # Calculer moyenne et std
    mean_val = df[value_col].mean()
    std_val = df[value_col].std()
    
    if std_val == 0:
        return []
    
    # Calculer Z-score
    df['zscore'] = np.abs((df[value_col] - mean_val) / std_val)
    
    # Filtrer les anomalies
    anomalies = df[df['zscore'] > threshold][
        [date_col, value_col, 'zscore']
    ].sort_values('zscore', ascending=False).to_dict('records')
    
    for anom in anomalies:
        if isinstance(anom[date_col], pd.Timestamp):
            anom[date_col] = anom[date_col].strftime('%Y-%m-%d')
        anom['zscore'] = round(anom['zscore'], 2)
    
    logger.info(f"✅ {len(anomalies)} anomalies détectées")
    return anomalies


def detect_emerging_hotspots(df: pd.DataFrame, date_col: str, location_col: str,
                            min_recent_count: int = 3) -> List[Dict]:
    """
    Détecte les micro-hotspots émergents (peu de données mais croissance).
    
    Args:
        df: DataFrame
        date_col: Colonne de dates
        location_col: Colonne de localisation
        min_recent_count: Seuil minimum d'occurrences récentes
    
    Returns:
        Liste de hotspots émergents
    """
    
    logger.info("🌱 Détection des hotspots émergents...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, location_col])
    
    # Diviser en deux périodes : récente et antérieure
    median_date = df[date_col].median()
    df_recent = df[df[date_col] >= median_date]
    df_old = df[df[date_col] < median_date]
    
    # Compter par localisation
    recent_counts = df_recent[location_col].value_counts()
    old_counts = df_old[location_col].value_counts()
    
    emerging = []
    for loc in recent_counts.index:
        recent_count = recent_counts[loc]
        old_count = old_counts.get(loc, 0)
        
        # Signal faible : peu avant, augmentation récente
        if recent_count >= min_recent_count and old_count < min_recent_count:
            emerging.append({
                'location': str(loc),
                'recent_count': int(recent_count),
                'old_count': int(old_count),
                'growth': int(recent_count - old_count),
                'recommendation': f"🌱 Micro-hotspot émergent à {loc}: {recent_count} cas récents → À surveiller."
            })
    
    emerging = sorted(emerging, key=lambda x: x['recent_count'], reverse=True)
    
    logger.info(f"✅ {len(emerging)} hotspots émergents détectés")
    return emerging
