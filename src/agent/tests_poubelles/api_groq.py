# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:47:10 2026

@author: amigo
"""

import os
import asyncio
import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.groq import Groq 
import time

load_dotenv()
nest_asyncio.apply()

df_311 = pd.read_csv("../../data/raw/requetes_311.csv", low_memory=False)
df_coll = pd.read_csv("../../data/raw/collisions_clean.csv")
df_meteo = pd.read_csv("../../data/raw/weather_montreal.csv")

llm = Groq(
    model="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    max_retries=5 # Augmente le nombre de tentatives en cas de 429
)

instruction_stricte = """\
You are an expert Pandas developer.
Your task: Output a SINGLE line of Python code that works with a DataFrame named `df`.
Constraint 1: The output must be a valid Python expression (one-liner).
Constraint 2: DO NOT assign the result to a variable (NO 'df = ...').
Constraint 3: Output ONLY the code. No explanations, no markdown, no triple backticks.

Example for counting: df[df['ACTI_NOM'] == 'Nid-de-poule'].shape[0]
Example for sum: df['NB_MORTS'].sum()
"""

engine_311 = PandasQueryEngine(df=df_311, llm=llm, instruction_str=instruction_stricte)
engine_coll = PandasQueryEngine(df=df_coll, llm=llm, instruction_str=instruction_stricte)
engine_meteo = PandasQueryEngine(df=df_meteo, llm=llm, instruction_str=instruction_stricte)

outil_311 = QueryEngineTool(
    query_engine=engine_311,
    metadata=ToolMetadata(
        name="donnees_311",
        description="Obligatoire pour les requêtes 311. La colonne pour les catégories est 'ACTI_NOM'. Pour les nids-de-poule, filtre avec la valeur exacte 'Nid-de-poule' et compte les lignes."
    )
)

outil_coll = QueryEngineTool(
    query_engine=engine_coll,
    metadata=ToolMetadata(
        name="donnees_collisions",
        description="Obligatoire pour les accidents. Pour trouver le nombre de morts, tu DOIS faire la somme de la colonne 'NB_MORTS'."
    )
)

outil_meteo = QueryEngineTool(
    query_engine=engine_meteo,
    metadata=ToolMetadata(
        name="donnees_meteo",
        description="Obligatoire pour obtenir les mesures exactes de température ou de précipitations."
    )
)

agent_config = ReActAgent(
    name="agent_donnees",
    description="Agent strict pour extraire des statistiques sur la mobilité.",
    system_prompt="Tu es un analyste de données strict. Tu utilises tes outils un par un. Ne génère JAMAIS de texte à trous. Attends le retour numérique de l'outil avant de formuler ta réponse finale.",
    tools=[outil_311, outil_coll, outil_meteo],
    llm=llm,
    max_iterations=4
)

workflow = AgentWorkflow(agents=[agent_config], root_agent="agent_donnees")

async def main():
    user_prompt = "Combien d'accidents y a-t-il eu les jours ou il y a eu plus de 10cm de neige"
    #"Cherche d'abord le nombre exact de 'Nid-de-poule' dans le dataset 311. Ensuite, calcule la somme des morts dans le dataset des collisions. Donne-moi les deux chiffres exacts."
    reponse = await workflow.run(user_msg=user_prompt)
    print(str(reponse))




tests_evaluation = {
    "Combien de morts au total dans les collisions ?": 269,
    "Nombre de requêtes pour Nid-de-poule ?": 112791,
    "Combien de nids-de-poule ont un statut 'Terminé' ?":95963,
    "Combien de nids-de-poule ont été réparé ?":95963,
    "Combien d'accidents y a-t-il eu les jours ou il y a eu plus de 10cm de neige": 1875,
    "Combien d'accidents y a-t-il eu de moins en 2020 par rapport à 2013 ": 18321
}

async def evaluer_agent(tests):
    resultats_logs = []
    
    for question, valeur_attendue in tests.items():
        print(f"Évaluation : {question}")
        
        # On demande explicitement le chiffre seul à l'agent
        prompt_brut = f"{question} Réponds UNIQUEMENT avec le nombre, sans texte."
        
        try:
            reponse = await workflow.run(user_msg=prompt_brut)
            valeur_calculee = str(reponse).strip()
            
            # Stockage pour analyse
            resultats_logs.append({
                "question": question,
                "attendu": valeur_attendue,
                "obtenu": valeur_calculee,
                "succes": str(valeur_attendue) in valeur_calculee
            })
            
            # Petit délai pour le quota Groq
            await asyncio.sleep(2) 
            
        except Exception as e:
            print(f"Erreur sur '{question}': {e}")

    return resultats_logs

# Dans ton bloc main :
if __name__ == "__main__":
    resultats = asyncio.run(evaluer_agent(tests_evaluation))
    
    # Affichage sous forme de table pour ton rapport
    df_eval = pd.DataFrame(resultats)
    print("\n--- RAPPORT D'ÉVALUATION ---")
    print(df_eval)
"""
if __name__ == "__main__":
    asyncio.run(main())
"""    