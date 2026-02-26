# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 22:22:02 2026

@author: amigo
"""

import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.ollama import Ollama

df_311 = pd.read_csv("../../data/raw/requetes_311_all_time.csv")

llm = Ollama(model="qwen2.5", request_timeout=600.0)

instruction_stricte = """\
You are working with a pandas DataFrame named `df`.
1. Convert the query to executable Python code using pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code must ALWAYS start with the variable `df` (e.g., `df['column'].value_counts()`).
4. PRINT ONLY THE EXPRESSION.
5. DO NOT RETURN ANY TEXT, EXPLANATIONS, OR MARKDOWN. ONLY THE RAW PYTHON EXPRESSION.
"""

query_engine = PandasQueryEngine(
    df=df_311, 
    llm=llm, 
    instruction_str=instruction_stricte, 
    verbose=True
)

reponse = query_engine.query("Quel est le top 3 des catégories de requêtes les plus fréquentes ? La colonne contenant les catégories est ACTI_NOM.")
print(reponse)