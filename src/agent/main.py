import os
import asyncio
import nest_asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging

from llama_index.llms.groq import Groq
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

load_dotenv()
nest_asyncio.apply()

# --- CONFIGURATION DU LOGGING ---
def setup_logging():
    """Configure le système de logging pour tracer l'exécution du agent"""
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "agent.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("🚀 DÉMARRAGE DU SYSTÈME D'ANALYSE MONTRÉAL")
    logger.info("=" * 80)
    return logger

logger = setup_logging()

from pydantic import Field
import time

class RateLimitedGroq(Groq):
    # On déclare le champ pour que Pydantic l'accepte
    delay_seconds: float = Field(default=2.0, description="Délai entre les appels API")

    def __init__(self, delay_seconds: float = 2.0, **kwargs):
        # On passe delay_seconds au constructeur de super()
        super().__init__(delay_seconds=delay_seconds, **kwargs)

    async def achat(self, messages, **kwargs):
        await asyncio.sleep(self.delay_seconds)
        return await super().achat(messages, **kwargs)

    async def acomplete(self, prompt, **kwargs):
        await asyncio.sleep(self.delay_seconds)
        return await super().acomplete(prompt, **kwargs)

    def chat(self, messages, **kwargs):
        time.sleep(self.delay_seconds)
        return super().chat(messages, **kwargs)

# --- CONFIGURATION ET CHARGEMENT ---
def setup_agent():
    # Chemin absolu au répertoire de données (depuis src/agent/main.py, remonte 2 niveaux)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    
    logger.info(f"📂 Chemin du projet: {PROJECT_ROOT}")
    logger.info(f"📊 Chemin des données: {DATA_DIR}")
    
    # Chargement des données
    logger.info("📥 Chargement des données...")
    df_311 = pd.read_csv(DATA_DIR / "requetes311.csv", low_memory=False)
    df_coll = pd.read_csv(DATA_DIR / "collisions.csv")
    df_meteo = pd.read_csv(DATA_DIR / "weather_montreal.csv")
    
    logger.info(f"✅ 311: {len(df_311)} requêtes chargées")
    logger.info(f"✅ Collisions: {len(df_coll)} collision chargées")
    logger.info(f"✅ Météo: {len(df_meteo)} enregistrements chargés")

    llm = RateLimitedGroq(model="llama-3.1-8b-instant", temperature=0.0, delay_seconds=2.0)
    logger.info("🤖 LLM Groq initialisé (llama-3.1-8b-instant)")

    instruction_stricte = """Output a SINGLE line of Python code using 'df'. 
    No 'df = ...', no markdown. Just the expression."""

    # Moteurs
    e_311 = PandasQueryEngine(df=df_311, llm=llm, instruction_str=instruction_stricte)
    e_coll = PandasQueryEngine(df=df_coll, llm=llm, instruction_str=instruction_stricte)
    e_meteo = PandasQueryEngine(df=df_meteo, llm=llm, instruction_str=instruction_stricte)
    
    logger.info("⚙️  Moteurs Pandas Query Engine initialisés")


    engines = {"311": e_311, "coll": e_coll, "meteo": e_meteo}
    
    
    tools = [
        QueryEngineTool(
            query_engine=e_311, 
            metadata=ToolMetadata(
                name="donnees_311",
                description="""Contient les requêtes citoyennes envoyées au 311 à Montréal.
                Colonnes clés : 
                - ACTI_NOM : Nom de l'activité. Valeurs exactes à utiliser pour filtrer : 'Nid-de-poule', 'Déneigement', 'Supa-Achat'.
                - DDS_DATE_CREATION : Date de création de la demande (format 'YYYY-MM-DD').
                - DERNIER_STATUT : Statut de la requête (ex: 'Terminée', 'Annulée').
                - DATE_DERNIER_STATUT : date de la dernière mise à jour
                - LOC_LONG, LOC_LAT : Coordonnées géographiques.
                Utilise cet outil pour quantifier les problèmes signalés par les citoyens (notamment les nids-de-poule) ou analyser les délais de traitement via DATE_DERNIER_STATUT."""
            )
        ),

        QueryEngineTool(
            query_engine=e_coll,
            metadata=ToolMetadata(
                name="donnees_collisions", 
                description="""Base de données des accidents et collisions routières à Montréal. 
                IMPORTANT : 
                    - Chaque ligne du DataFrame représente UN SEUL accident. 
                    - Pour compter le nombre d'accidents, utilise len(df) ou .shape[0].
                    - Ne sommez PAS NB_VICTIMES_TOTAL sauf si on demande spécifiquement le nombre de blessés.
                Colonnes clés : 
                - DATE: Date de l'accident (format 'YYYY-MM-DD'). Utilise cette colonne pour filtrer par année ou par mois.
                - GRAVITE : Gravité de l'accident ('Dégâts matériels seulement', 'Léger', 'Grave', 'Mortel').
                - NB_MORTS, NB_VICTIMES_TOTAL, NB_BLESSES_GRAVES : Nombre de victimes (Colonnes numériques).
                - LOC_LAT, LOC_LONG : Coordonnées géographiques.
                Utilise cet outil pour calculer des bilans de sécurité routière, identifier des zones accidentogènes ou comparer la mortalité entre différentes années."""
            )
        ),

        QueryEngineTool(
            query_engine=e_meteo,
            metadata=ToolMetadata(
                name="donnees_meteo",
                description="""Historique météorologique quotidien de Montréal.
                Colonnes clés :
                - DATE : Date de l'observation (format 'YYYY-MM-DD'). Pivot central pour les jointures avec les autres outils.
                - temperature_2m_max, temperature_2m_min : Températures extrêmes de la journée (°C).
                - precipitation_sum : Quantité totale de pluie (en mm).
                - snowfall_sum : Quantité totale de neige (en cm).
                Utilise cet outil pour corréler les conditions climatiques (tempêtes de neige, verglas) avec le volume de requêtes 311 ou le nombre d'accidents de la route."""
            )
        )
    
    ]

    system_prompt = """Tu es un analyste de données expert pour Montréal. Tu as accès à trois bases de données sur les requêtes 311, les collisions routières et les données météorologiques.

Pour chaque question :
1. Identifie quelle(s) base(s) de données utiliser
2. Formule la requête en Python Pandas
3. Exécute l'analyse
4. Donne une réponse chiffrée et factuelle

Bases disponibles :
- donnees_311 : requêtes citoyennes avec motifs, dates et statuts
- donnees_collisions : accidents avec gravité, victimes et localisation
- donnees_meteo : conditions météo quotidiennes

Sois précis, chiffré et objectif dans tes réponses."""

    return llm, engines, tools, system_prompt


# --- LOGIQUE D'ÉVALUATION ---
async def run_benchmarks(llm, system_prompt):
    """Fonction de test pour valider les réponses."""
    logger.info("=" * 80)
    logger.info("📊 DÉMARRAGE DES BENCHMARKS")
    logger.info("=" * 80)
    
    tests_evaluation = {
        "Combien de morts au total dans les collisions ?": 269,
        "Nombre de requêtes pour Nid-de-poule ?": 112791,
        "Combien d'accidents y a-t-il eu les jours ou il y a eu plus de 10cm de neige": 1875
    }

    resultats_logs = []
    
    for i, (question, attendu) in enumerate(tests_evaluation.items(), 1):
        logger.info(f"\n📌 TEST {i}/{len(tests_evaluation)}: {question}")
        print(f"\n Test : {question}")
        try:
            reponse_finale = await run(llm, question, system_prompt)
            succes = str(attendu) in reponse_finale
            logger.info(f"✅ Attention: {attendu} | Obtenu: {reponse_finale[:100]}")
            
            resultats_logs.append({
                "Question": question,
                "Attendu": attendu,
                "Obtenu": reponse_finale,
                "Succès": succes
            })
        except Exception as e:
            logger.error(f"❌ Erreur sur ce test : {e}")
            print(f"❌ Erreur sur ce test : {e}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ BENCHMARKS TERMINÉS")
    logger.info("=" * 80)
    return pd.DataFrame(resultats_logs)

async def run(llm, prompt: str, engines: dict, system_prompt: str = "") -> str:
    """
    Appel à l'API Groq avec accès réel aux données via les PandasQueryEngines.
    
    Args:
        llm: Objet Groq configuré
        prompt: Question/prompt de l'utilisateur
        engines: Dict contenant les PandasQueryEngines {"311": engine, "coll": engine, "meteo": engine}
        system_prompt: Instructions systèmes pour le LLM
        
    Returns:
        Réponse textuelle de Groq avec résultats réels des bases de données
    """
    logger.info(f"🔵 RUN: Question reçue: {prompt[:80]}...")
    
    try:
        # Étape 1 : Groq analyse la question et choisit le bon engine
        analysis_prompt = f"""{system_prompt}

Question utilisateur: {prompt}

Réponds EXACTEMENT le nom du moteur à utiliser parmi : donnees_311, donnees_collisions, donnees_meteo"""
        
        engine_choice_response = llm.complete(analysis_prompt)
        engine_choice = str(engine_choice_response.text).strip().lower() if hasattr(engine_choice_response, 'text') else ""
        logger.info(f"🎯 Engine sélectionné: {engine_choice}")
        
        # Étape 2 : Mapper le choix au bon engine
        engine_map = {
            "donnees_311": engines["311"],
            "donnees_collisions": engines["coll"],
            "donnees_meteo": engines["meteo"]
        }
        
        # Trouver l'engine approprié
        selected_engine = None
        for key in engine_map:
            if key in engine_choice:
                selected_engine = engine_map[key]
                break
        
        # Si pas d'engine trouvé, utiliser celui de la météo par défaut
        if selected_engine is None:
            selected_engine = engines["meteo"]
            logger.warning("⚠️  Engine par défaut utilisé (météo)")
        
        # Étape 3 : Exécuter la requête sur les données réelles
        logger.info("📊 Exécution de la requête sur les données...")
        try:
            data_response = selected_engine.query(prompt)
            data_result = str(data_response).strip()
            logger.info(f"✅ Résultat brut: {data_result[:100]}...")
        except Exception as e:
            data_result = f"Erreur lors de l'exécution sur les données : {str(e)}"
            logger.error(f"❌ Erreur query engine: {str(e)}")
        
        # Étape 4 : Groq formule la réponse finale avec les données réelles
        logger.info("🧠 Formulation de la réponse finale...")
        final_prompt = f"""{system_prompt}

Données brutes obtenues : {data_result}

Question originale : {prompt}

Formule une réponse factuelle basée sur les données obtenues. Sois précis et objectif."""
        
        final_response = llm.complete(final_prompt)
        reponse = str(final_response.text).strip() if hasattr(final_response, 'text') else str(final_response).strip()
        logger.info(f"✅ Réponse finale générée: {reponse[:100]}...")
        
        return reponse
        
    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE: {str(e)}")
        return f"Erreur lors de l'appel à Groq : {str(e)}"


if __name__ == "__main__":
    # Test simple
    logger.info("\n" + "=" * 80)
    logger.info("🎯 MODE TEST SIMPLE")
    logger.info("=" * 80)
    
    llm, engines, tools, system_prompt = setup_agent()
    question = "Nombre de jours en dessous de 0 degrés à Montréal"
    logger.info(f"❓ Question: {question}")
    
    reponse = asyncio.run(run(llm, question, engines, system_prompt))
    logger.info(f"📋 Réponse: {reponse}")
    
    print(f"\nQuestion: {question}")
    print(f"Réponse: {reponse}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TEST TERMINÉ")
    logger.info("=" * 80)