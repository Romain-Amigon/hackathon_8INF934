import os
import asyncio
import nest_asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

load_dotenv()
nest_asyncio.apply()

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
    
    # Chargement des données
    df_311 = pd.read_csv(DATA_DIR / "requetes311.csv", low_memory=False)
    df_coll = pd.read_csv(DATA_DIR / "collisions.csv")
    df_meteo = pd.read_csv(DATA_DIR / "weather_montreal.csv")

    llm = RateLimitedGroq(model="llama-3.1-8b-instant", temperature=0.0, delay_seconds=2.0)

    instruction_stricte = """Output a SINGLE line of Python code using 'df'. 
    No 'df = ...', no markdown. Just the expression."""

    # Moteurs
    e_311 = PandasQueryEngine(df=df_311, llm=llm, instruction_str=instruction_stricte)
    e_coll = PandasQueryEngine(df=df_coll, llm=llm, instruction_str=instruction_stricte)
    e_meteo = PandasQueryEngine(df=df_meteo, llm=llm, instruction_str=instruction_stricte)


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
    tests_evaluation = {
        "Combien de morts au total dans les collisions ?": 269,
        "Nombre de requêtes pour Nid-de-poule ?": 112791,
        "Combien d'accidents y a-t-il eu les jours ou il y a eu plus de 10cm de neige": 1875
    }

    resultats_logs = []
    
    for question, attendu in tests_evaluation.items():
        print(f"\n Test : {question}")
        try:
            reponse_finale = await run(llm, question, system_prompt)
            
            resultats_logs.append({
                "Question": question,
                "Attendu": attendu,
                "Obtenu": reponse_finale,
                "Succès": str(attendu) in reponse_finale
            })
        except Exception as e:
            print(f"❌ Erreur sur ce test : {e}")

    return pd.DataFrame(resultats_logs)

async def run(llm, prompt: str, system_prompt: str = "") -> str:
    """
    Appel simple à l'API Groq avec le prompt engineering.
    
    Args:
        llm: Objet Groq configuré
        prompt: Question/prompt de l'utilisateur
        system_prompt: Instructions systèmes pour le LLM
        
    Returns:
        Réponse textuelle de Groq
    """
    try:
        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}" if system_prompt else prompt
        response = llm.complete(full_prompt)
        return str(response.text).strip() if hasattr(response, 'text') else str(response).strip()
    except Exception as e:
        return f"Erreur lors de l'appel à Groq : {str(e)}"


if __name__ == "__main__":
    # Test simple
    llm, engines, tools, system_prompt = setup_agent()
    question = "Combien d'accidents y a-t-il eu les jours ou il y a eu plus de 10cm de neige"
    reponse = asyncio.run(run(llm, question, system_prompt))
    print(f"Question: {question}")
    print(f"Réponse: {reponse}")