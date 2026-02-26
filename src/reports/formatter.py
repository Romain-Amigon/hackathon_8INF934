"""
Module pour formater les briefings en deux versions : public et municipalité.
"""

from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def format_briefing_public(briefing_data: Dict) -> str:
    """
    Formate un briefing pour le grand public.
    Langage simple, focus sur sécurité routière.
    """
    
    md = []
    md.append("# 🚗 Sécurité Routière à Montréal")
    md.append(f"**Rapport du {briefing_data['generated_at'].split()[0]}** · {briefing_data['period']}")
    md.append("")
    
    # KPIs principaux
    md.append("## 📊 Résumé")
    md.append(f"- **{briefing_data['total_collisions']:,}** accidents signalés")
    md.append(f"- **{briefing_data['total_311']:,}** demandes de maintenance")
    md.append("")
    
    # Top hotspots
    md.append("## ⚠️ Zones à Risque")
    for i, spot in enumerate(briefing_data['hotspots_collisions'][:3], 1):
        md.append(f"### {i}. Latitude {spot['center_lat']:.2f}, Longitude {spot['center_lon']:.2f}")
        md.append(f"- **{spot['count']} accidents** ({spot['pct_graves']:.0f}% graves)")
        if spot['morts'] > 0:
            md.append(f"- ⚠️ **{spot['morts']} mort(s)**, {spot['blesses_graves']} blessés graves")
        md.append("")
    
    # Maintenance
    md.append("## 🔧 Types de Maintenance Urgents")
    for i, item in enumerate(briefing_data['hotspots_311'][:3], 1):
        md.append(f"**{i}. {item['motif']}** ({item['arrondissement']})")
        md.append(f"   - {item['count']} demandes ({item['pct_of_total']:.0f}% du total)")
    md.append("")
    
    # Signaux faibles
    if briefing_data.get('weak_signals'):
        md.append("## 🔔 À Surveiller")
        for signal in briefing_data['weak_signals']:
            md.append(f"- {signal['recommendation']}")
        md.append("")
    
    # Recommandations
    if briefing_data.get('recommendations'):
        md.append("## 💡 Conseils aux Citoyens")
        for rec in briefing_data['recommendations'][:3]:
            if '🚨' in rec:
                md.append(f"- {rec}")
        md.append("")
    
    md.append("---")
    md.append("*Données de la Ville de Montréal · Mise à jour automatique*")
    
    return "\n".join(md)


def format_briefing_municipality(briefing_data: Dict) -> str:
    """
    Formate un briefing pour les autorités municipales.
    Langage technique, focus sur KPIs opérationnels.
    """
    
    md = []
    md.append("# 📋 Briefing Opérationnel Mobilité")
    md.append(f"**Généré le {briefing_data['generated_at']}**")
    md.append(f"Période: {briefing_data['period']}")
    md.append("")
    
    # Métriques (KPIs)
    md.append("## 📈 KPIs Principaux")
    total_coll = briefing_data['total_collisions']
    total_311 = briefing_data['total_311']
    
    md.append(f"| Métrique | Valeur | Tendance |")
    md.append("|----------|--------|----------|")
    md.append(f"| Collisions | {total_coll:,} | - |")
    md.append(f"| Requêtes 311 | {total_311:,} | - |")
    
    # Tendances YoY si disponibles
    if briefing_data.get('trends_yoy'):
        for period, trend in list(briefing_data['trends_yoy'].items())[-1:]:
            md.append(f"| Tendance YoY | {trend['change_pct']:+.1f}% | {trend['direction']} |")
    md.append("")
    
    # Hotspots opérationnels (coordonnées précises)
    md.append("## 🎯 Hotspots Prioritaires (Interventions Requises)")
    for i, spot in enumerate(briefing_data['hotspots_collisions'][:3], 1):
        md.append(f"\n### Hotspot #{i}")
        md.append(f"**Localisation**: {spot['center_lat']:.4f}, {spot['center_lon']:.4f}")
        md.append(f"**Incidents**: {spot['count']} ({spot['graves']} graves)")
        md.append(f"**Pertes humaines**: {spot['morts']} morts, {spot['blesses_graves']} blessés graves")
        md.append(f"**Action recommandée**: Inspection immédiate + Renforcement signalisation")
    md.append("")
    
    # 311 par catégorie
    md.append("## 🔧 Charges Maintenance par Catégorie")
    for item in briefing_data['hotspots_311'][:5]:
        md.append(f"- **{item['motif']}** ({item['arrondissement']}): {item['count']} → Priorité {item['priority']}")
    md.append("")
    
    # Signaux faibles (alerte opérationnelle)
    if briefing_data.get('weak_signals'):
        md.append("## 🚨 Signaux Faibles Détectés (Alerte Préventive)")
        for signal in briefing_data['weak_signals']:
            md.append(f"\n**{signal['category']}**")
            md.append(f"- Pente de croissance: {signal['slope']:.2f} inc/sem")
            md.append(f"- Trend strength: +{signal['trend_strength']:.0f}% sur 6 sem")
            md.append(f"- Significativité: p={signal['p_value']:.4f}")
            md.append(f"- Action: {signal['recommendation']}")
    md.append("")
    
    # Recommandations opérationnelles
    if briefing_data.get('recommendations'):
        md.append("## 📌 Recommandations d'Action")
        for rec in briefing_data['recommendations']:
            md.append(f"- {rec}")
    md.append("")
    
    md.append("---")
    md.append("*Confidentialité Municipale · Briefing Automatisé*")
    
    return "\n".join(md)


def format_briefing_csv_export(briefing_data: Dict) -> str:
    """
    Exporte les données du briefing en CSV pour intégration dans SI.
    """
    
    csv_lines = []
    
    # En-têtes
    csv_lines.append("type,category,value,unit,date_generated")
    
    # Hotspots collisions
    for i, spot in enumerate(briefing_data['hotspots_collisions'], 1):
        csv_lines.append(f"hotspot_collision,zone_{i},{spot['count']},count,{briefing_data['generated_at']}")
        csv_lines.append(f"hotspot_collision,zone_{i}_graves,{spot['graves']},count,{briefing_data['generated_at']}")
    
    # Hotspots 311
    for item in briefing_data['hotspots_311']:
        csv_lines.append(f"hotspot_311,{item['motif']},{item['count']},count,{briefing_data['generated_at']}")
    
    # Signaux faibles
    for signal in briefing_data.get('weak_signals', []):
        csv_lines.append(f"weak_signal,{signal['category']},{signal['trend_strength']},pct_change,{briefing_data['generated_at']}")
    
    return "\n".join(csv_lines)
