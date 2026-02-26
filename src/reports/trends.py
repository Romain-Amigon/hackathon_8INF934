"""
Module pour analyser les tendances temporelles : YoY, MoM, etc.
"""

import pandas as pd
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def analyze_trends_yoy(df: pd.DataFrame, date_col: str, value_col: str = None) -> Dict:
    """
    Analyse les tendances année-sur-année.
    
    Args:
        df: DataFrame
        date_col: Colonne contenant les dates
        value_col: Colonne à compter (None = nombre de lignes)
    
    Returns:
        Dict avec tendances YoY, % changement
    """
    
    logger.info(f"📊 Analyse YoY sur {date_col}...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    if len(df) == 0:
        return {}
    
    df['YEAR'] = df[date_col].dt.year
    
    # Agrégation par année
    if value_col:
        yearly = df.groupby('YEAR')[value_col].sum()
    else:
        yearly = df.groupby('YEAR').size()
    
    # Calculer % changement YoY
    trends = {}
    for i in range(len(yearly) - 1):
        year_curr = yearly.index[i + 1]
        year_prev = yearly.index[i]
        val_curr = yearly.iloc[i + 1]
        val_prev = yearly.iloc[i]
        
        if val_prev == 0:
            pct_change = float('inf') if val_curr > 0 else 0
        else:
            pct_change = (val_curr - val_prev) / val_prev * 100
        
        trends[f'{year_prev}_to_{year_curr}'] = {
            'year_prev': year_prev,
            'year_curr': year_curr,
            'value_prev': int(val_prev),
            'value_curr': int(val_curr),
            'change_abs': int(val_curr - val_prev),
            'change_pct': round(pct_change, 1),
            'direction': '↑' if pct_change > 0 else '↓' if pct_change < 0 else '→'
        }
    
    logger.info(f"✅ Tendances YoY analysées: {len(trends)} périodes")
    return trends


def analyze_trends_mom(df: pd.DataFrame, date_col: str, value_col: str = None,
                       last_n_months: int = 12) -> List[Dict]:
    """
    Analyse les tendances mois-sur-mois (derniers N mois).
    
    Args:
        df: DataFrame
        date_col: Colonne contenant les dates
        value_col: Colonne à compter (None = nombre de lignes)
        last_n_months: Nombre de mois à analyser
    
    Returns:
        Liste de dicts : {month, value, change_pct}
    """
    
    logger.info(f"📊 Analyse MoM sur {last_n_months} mois...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    if len(df) == 0:
        return []
    
    # Créer période mensuelle
    df['MONTH'] = df[date_col].dt.to_period('M')
    
    # Agrégation par mois
    if value_col:
        monthly = df.groupby('MONTH')[value_col].sum()
    else:
        monthly = df.groupby('MONTH').size()
    
    # Garder les N derniers mois
    monthly = monthly.tail(last_n_months)
    
    # Calculer % changement MoM
    trends = []
    for i in range(len(monthly)):
        val_curr = monthly.iloc[i]
        val_prev = monthly.iloc[i - 1] if i > 0 else None
        
        if val_prev is None:
            pct_change = 0
        elif val_prev == 0:
            pct_change = float('inf') if val_curr > 0 else 0
        else:
            pct_change = (val_curr - val_prev) / val_prev * 100
        
        trends.append({
            'month': str(monthly.index[i]),
            'value': int(val_curr),
            'value_prev': int(val_prev) if val_prev is not None else None,
            'change_pct': round(pct_change, 1) if val_prev is not None else 0,
            'direction': '↑' if pct_change > 0 else '↓' if pct_change < 0 else '→'
        })
    
    logger.info(f"✅ Tendances MoM analysées: {len(trends)} mois")
    return trends


def analyze_trends_by_category(df: pd.DataFrame, date_col: str, category_col: str,
                               top_n: int = 5) -> Dict[str, List[Dict]]:
    """
    Analyse les tendances MoM par catégorie (ex: ACTI_NOM par mois).
    
    Args:
        df: DataFrame
        date_col: Colonne contenant les dates
        category_col: Colonne de catégories
        top_n: Top N catégories à analyser
    
    Returns:
        Dict : {categorie: [liste de mois avec valeurs]}
    """
    
    logger.info(f"📊 Analyse des tendances par {category_col}...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df[category_col] = df[category_col].fillna('Inconnu')
    
    # Top N catégories
    top_cats = df[category_col].value_counts().head(top_n).index.tolist()
    
    # Créer période mensuelle
    df['MONTH'] = df[date_col].dt.to_period('M')
    
    # Agrégation par mois + catégorie
    trends_by_cat = {}
    for cat in top_cats:
        df_cat = df[df[category_col] == cat]
        monthly = df_cat.groupby('MONTH').size()
        
        trends_by_cat[cat] = [
            {
                'month': str(month),
                'value': int(monthly[month]),
                'pct_of_total': round(100 * monthly[month] / df[df['MONTH'] == month].shape[0], 1)
            }
            for month in monthly.index
        ]
    
    logger.info(f"✅ Tendances par catégorie analysées: {len(trends_by_cat)} catégories")
    return trends_by_cat


def detect_peak_periods(df: pd.DataFrame, date_col: str, hour_col: str = None) -> Dict:
    """
    Détecte les heures/périodes de pic.
    
    Args:
        df: DataFrame
        date_col: Colonne de dates
        hour_col: Colonne d'heures (optionnel, extrait de date_col sinon)
    
    Returns:
        Dict : {peak_hour, peak_day_of_week, etc.}
    """
    
    logger.info("📊 Détection des pics horaires...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    if hour_col is None:
        df['HOUR'] = df[date_col].dt.hour
    else:
        # Essayer d'extraire l'heure
        df['HOUR'] = pd.to_datetime(df[hour_col], format='%H:%M:%S', errors='coerce').dt.hour
    
    df['DOW'] = df[date_col].dt.day_name()
    
    peaks = {
        'peak_hour': int(df['HOUR'].value_counts().idxmax()) if 'HOUR' in df.columns else None,
        'peak_day': str(df['DOW'].value_counts().idxmax()),
        'hourly_dist': df['HOUR'].value_counts().sort_index().to_dict() if 'HOUR' in df.columns else {}
    }
    
    logger.info(f"✅ Pics détectés: {peaks['peak_hour']}h, jour: {peaks['peak_day']}")
    return peaks
