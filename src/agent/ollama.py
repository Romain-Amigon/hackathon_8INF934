# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 23:37:06 2026

@author: amigo
"""

import asyncio
import nest_asyncio
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent

nest_asyncio.apply()

df_311 = pd.read_csv("../../data/raw/requetes_311.csv", low_memory=False)
df_coll = pd.read_csv("../../data/raw/collisions_clean.csv")
df_meteo = pd.read_csv("../../data/raw/weather_montreal.csv")

llm = Ollama(model="qwen2.5", request_timeout=300.0)

instruction_stricte = """\
You are working with a pandas DataFrame named `df`.
1. Convert the query to executable Python code using pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code must ALWAYS start with the variable `df`.
4. PRINT ONLY THE EXPRESSION.
5. DO NOT RETURN ANY TEXT, EXPLANATIONS, OR MARKDOWN. ONLY THE RAW PYTHON EXPRESSION.
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
    user_prompt = "Cherche d'abord le nombre exact de 'Nid-de-poule' dans le dataset 311. Ensuite, calcule la somme des morts dans le dataset des collisions. Donne-moi les deux chiffres exacts."
    reponse = await workflow.run(user_msg=user_prompt)
    print(str(reponse))

if __name__ == "__main__":
    asyncio.run(main())