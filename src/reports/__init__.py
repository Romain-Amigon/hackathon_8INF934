"""
Module de rapports et synthèses sans RAG.
Contient : hotspots, tendances, signaux faibles, briefings.
"""

from .hotspots import get_top_hotspots_collisions, get_top_hotspots_311
from .trends import analyze_trends_yoy, analyze_trends_mom
from .weak_signals import detect_weak_signals
from .briefing import generate_briefing
from .formatter import format_briefing_public, format_briefing_municipality

__all__ = [
    'get_top_hotspots_collisions',
    'get_top_hotspots_311',
    'analyze_trends_yoy',
    'analyze_trends_mom',
    'detect_weak_signals',
    'generate_briefing',
    'format_briefing_public',
    'format_briefing_municipality',
]
