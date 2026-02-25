import os
import asyncio
import nest_asyncio
import pandas as pd
from dotenv import load_dotenv

from graph import create_graph
from nodes import Nodes

from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
load_dotenv()
nest_asyncio.apply()

from llama_index.core.base.llms.types import ChatMessage, CompletionResponse

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
    # Chargement des données
    df_311 = pd.read_csv("../../data/raw/requetes_311.csv", low_memory=False)
    df_coll = pd.read_csv("../../data/raw/collisions_clean.csv")
    df_meteo = pd.read_csv("../../data/raw/weather_montreal.csv")

    llm = Groq(model="llama-3.1-8b-instant", temperature=0.0)

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

    return ReActAgent(
        name="agent_donnees",
        description="Agent strict pour extraire des statistiques sur la mobilité.",
        system_prompt="Tu es un analyste de données strict. Tu utilises tes outils un par un. Ne génère JAMAIS de texte à trous. Attends le retour numérique de l'outil avant de formuler ta réponse finale.",
        tools=tools,
        llm=llm,
        max_iterations=4
    ), engines


# --- LOGIQUE D'ÉVALUATION ---
async def run_benchmarks(app):
    tests_evaluation = {
        "Combien de morts au total dans les collisions ?": 269,
        "Nombre de requêtes pour Nid-de-poule ?": 112791,
        "Combien d'accidents y a-t-il eu les jours ou il y a eu plus de 10cm de neige": 1875
    }

    resultats_logs = []
    
    for question, attendu in tests_evaluation.items():
        print(f"\n Test : {question}")
        try:
            # On lance le GRAPH et non l'agent directement
            inputs = {"messages": [f"{question}. Réponds juste le chiffre."]}
            output = await app.ainvoke(inputs)
            
            # On récupère le dernier message du state
            reponse_finale = str(output["messages"][-1]).strip()
            
            resultats_logs.append({
                "Question": question,
                "Attendu": attendu,
                "Obtenu": reponse_finale,
                "Succès": str(attendu) in reponse_finale
            })
        except Exception as e:
            print(f"❌ Erreur sur ce test : {e}")

    return pd.DataFrame(resultats_logs)

async def run(app, prompt):
    inputs = {"messages": [prompt]}
    reponse_finale = "Une erreur est survenue."
    reflexions_finales = []

    try:
        output = await app.ainvoke(inputs)
        
        if "reflexions" in output:
            reflexions_finales = output["reflexions"]
        
        if "messages" in output and len(output["messages"]) > 0:
            dernier_msg = output["messages"][-1]
            
            if hasattr(dernier_msg, 'content'):
                reponse_finale = str(dernier_msg.content).strip()
            else:
                reponse_finale = str(dernier_msg).strip()
                
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")

    return reponse_finale, reflexions_finales

def setup_retriever(data_dir="../../data/rag", persist_dir="../../data/rag/vectors"):
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    if not os.path.exists(persist_dir):
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        
    return index.as_retriever(similarity_top_k=7)

# --- MAIN ---
async def main_benchmark():
    # 1. Initialiser l'agent de base
    agent_instance = setup_agent()
    
    # 2. Compiler le graphe LangGraph
    app = create_graph(agent_instance)
    
    # 3. Lancer l'évaluation
    df_report = await run_benchmarks(app)
    
    print("\n" + "="*30)
    print("RAPPORT D'ÉVALUATION FINAL")
    print("="*30)
    print(df_report)

# --- MAIN ---
async def main():
    # 1. Initialiser l'agent de base
    agent_instance,engines = setup_agent()
    retriever = setup_retriever()
    # 2. Compiler le graphe LangGraph
    app = create_graph(agent_instance,engines,retriever)
    
    #prompt =input("Bonjour, comment puis-je vous aider ?\n")
    prompt="Combien d'accidents y a-t-il eu les jours ou il y a eu plus de 10cm de neige"
    reponse,reflexions = await run(app, prompt)
    
    print('----------')
    
    print("\n" + "="*40)
    print("RÉFLEXIONS DU MODÈLE")
    print("="*40)
    for r in reflexions:
        print(f"\x1B[3m {r}\x1B[0m ")
        print("-" * 40)
        
    print("\n" + "="*40)
    print("RÉPONSE FINALE")
    print("="*40)
    print(reponse)
    time.sleep(3)
    prompt = "Quelle est la définition d'un accident grave ?"
    reponse , reflexions= await run(app, prompt)
    
    
    print('----------')
    
    print("\n" + "="*40)
    print("RÉFLEXIONS DU MODÈLE")
    print("="*40)
    for r in reflexions:
        print(f"\x1B[3m {r}\x1B[0m ")
        print("-" * 40)
        
    print("\n" + "="*40)
    print("RÉPONSE FINALE")
    print("="*40)
    print(reponse)
    
    
    
    time.sleep(3)
    
    prompt = "Quelle est la définition d'un accident grave ?"
    reponse , reflexions= await run(app, prompt)
    print('----------')
    
    print("\n" + "="*40)
    print("RÉFLEXIONS DU MODÈLE")
    print("="*40)
    for r in reflexions:
        print(f"\x1B[3m {r}\x1B[0m ")
        print("-" * 40)
        
    print("\n" + "="*40)
    print("RÉPONSE FINALE")
    print("="*40)
    print(reponse)

if __name__ == "__main__":
    asyncio.run(main())