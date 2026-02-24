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

load_dotenv()
nest_asyncio.apply()

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
        QueryEngineTool(query_engine=e_311, metadata=ToolMetadata(name="donnees_311", description="Requêtes 311")),
        QueryEngineTool(query_engine=e_coll, metadata=ToolMetadata(name="donnees_coll", description="Collisions")),
        QueryEngineTool(query_engine=e_meteo, metadata=ToolMetadata(name="donnees_meteo", description="Météo"))
    ]

    return ReActAgent(
        name="agent_donnees",
        description="Agent strict pour extraire des statistiques sur la mobilité.",
        system_prompt="Tu es un analyste de données strict. Tu utilises tes outils un par un. Ne génère JAMAIS de texte à trous. Attends le retour numérique de l'outil avant de formuler ta réponse finale.",
        tools=tools,
        llm=llm,
        max_iterations=3
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

    try:
        output = await app.ainvoke(inputs)
        
        if "messages" in output and len(output["messages"]) > 0:
            dernier_msg = output["messages"][-1]
            
            # Si c'est un objet (comme ce que tu as reçu), on prend .content
            if hasattr(dernier_msg, 'content'):
                reponse_finale = str(dernier_msg.content).strip()
            else:
                reponse_finale = str(dernier_msg).strip()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")

    return reponse_finale

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
    
    # 2. Compiler le graphe LangGraph
    app = create_graph(agent_instance,engines)
    
    #prompt =input("Bonjour, comment puis-je vous aider ?\n")
    prompt="Nombre de requêtes pour Nid-de-poule dans le dataset 311?"
    reponse = await run(app, prompt)
    
    print(reponse)

if __name__ == "__main__":
    asyncio.run(main())