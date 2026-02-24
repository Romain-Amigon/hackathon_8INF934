import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, inspect
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

# LlamaIndex / Ollama imports
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import torch
device = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "cpu"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device=device
)

Settings.llm = Ollama(
    model="qwen2.5-coder:7b", 
    base_url="http://127.0.0.1:11434", 
    request_timeout=600.0
)

# Connexion à la base de données fournie
db_path = "/app/data/raw/mobility.db"
engine = create_engine(f"sqlite:///{db_path}")

# --- 2. UTILITAIRES DE SCHÉMA (Critère : Vocabulaire spécifique) ---
def get_real_schema():
    """
    Extrait la structure réelle des tables pour adapter le système 
    au domaine spécifique de la mobilité[cite: 56].
    """
    inst = inspect(engine)
    schema_info = ""
    for table in ["requetes_311", "collisions", "weather_montreal"]:
        try:
            cols = [c['name'] for c in inst.get_columns(table)]
            schema_info += f"- Table '{table}': Colonnes {cols}\n"
        except Exception:
            continue
    return schema_info

# --- 3. DÉFINITION DE L'AGENT LANGGRAPH (Critère : Résilience) ---

class AgentState(TypedDict):
    question: str
    sql_query: str
    results: str
    error: str
    retry_count: int
    final_response: str

def get_detailed_schema():
    """Schéma ultra-détaillé pour verrouiller le LLM (Critère CdC: Précision)"""
    return """
    TABLES ET COLONNES RÉELLES (SQLite) :
    1. Table 'requetes_311':
       - Colonnes: ['ID_UNIQUE', 'ACTI_NOM', 'ARRONDISSEMENT', 'DDS_DATE_CREATION', 'LOC_LAT', 'LOC_LONG']
       - Note: Pour compter les requêtes, utilisez COUNT(ID_UNIQUE).
    
    2. Table 'collisions':
       - Colonnes: ['DT_ACCDN', 'GRAVITE', 'NB_VICTIMES_TOTAL', 'LOC_LAT', 'LOC_LONG']
       - Note: IL N'Y A PAS DE COLONNE 'ID'. Pour compter les collisions, utilisez COUNT(*).
       - GRAVITE est une chaîne (ex: 'Grave', 'Léger', 'Mortel'). Ne pas utiliser > 0.
    
    3. Table 'weather_montreal':
       - Colonnes: ['time', 'precipitation_sum', 'snowfall_sum']
       - Note: 'time' est une date. Jointure avec collisions sur date(DT_ACCDN) = date(time).
    """

def generate_sql_node(state: AgentState):
    schema = get_detailed_schema()
    
    prompt = f"""Tu es un expert SQL SQLite pour Montréal.
    INTERDICTION : Ne jamais utiliser la colonne 'ID', elle n'existe pas.
    
    {schema}
    
    EXEMPLES DE RÉFÉRENCE :
    - Compter collisions: SELECT COUNT(*) FROM collisions
    - Filtrer neige: WHERE snowfall_sum > 0
    - Joindre géo: ON round(c.LOC_LAT, 3) = round(r.LOC_LAT, 3)
    
    QUESTION : {state['question']}
    SORTIE : SQL UNIQUEMENT (SANS TEXTE, SANS BALISES)
    """
    
    if state['error']:
        prompt += f"\n ERREUR À CORRIGER : {state['error']}. Rappel: PAS de colonne 'ID'."

    response = Settings.llm.complete(prompt)
    sql = response.text.replace("```sql", "").replace("```", "").strip()
    
    # Nettoyage ultime : On prend tout entre le premier SELECT et le premier point-virgule
    import re
    match = re.search(r"(SELECT.*;)", sql, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1)
        
    return {"sql_query": sql, "retry_count": state['retry_count'] + 1}

def execute_sql_node(state: AgentState):
    """Valide et exécute la requête (Composant Validator)[cite: 57, 63]."""
    try:
        df = pd.read_sql(state['sql_query'], engine)
        return {"results": df.to_string(), "error": None}
    except Exception as e:
        return {"error": str(e)}

def finalize_node(state: AgentState):
    """Produit la synthèse finale pour l'utilisateur[cite: 23, 67]."""
    prompt = f"""Basé sur ces données :
    {state['results']}
    Réponds à la question : {state['question']}
    Sois précis et cite les chiffres.
    """
    response = Settings.llm.complete(prompt)
    return {"final_response": response.text}

def should_continue(state: AgentState):
    """Logique de contrôle pour la résilience aux erreurs."""
    if state['error'] and state['retry_count'] < 3:
        return "continue"
    return "end"

# --- 4. CONSTRUCTION DU WORKFLOW ---
workflow = StateGraph(AgentState)

workflow.add_node("sql_gen", generate_sql_node)
workflow.add_node("execute", execute_sql_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("sql_gen")
workflow.add_edge("sql_gen", "execute")

workflow.add_conditional_edges(
    "execute",
    should_continue,
    {
        "continue": "sql_gen",
        "end": "finalize"
    }
)

workflow.add_edge("finalize", END)
app_langgraph = workflow.compile()

# --- 5. EXÉCUTION (Démonstration fonctionnelle) ---
def main():
    print(" Lancement de l'agent Mobility Copilot (LangGraph)...")
    
    inputs = {
        "question": "Quels secteurs ont une hausse de collisions en conditions de pluie/neige ?",
        "sql_query": "",
        "results": "",
        "error": None,
        "retry_count": 0,
        "final_response": ""
    }

    # Stream des étapes pour la démonstration [cite: 71]
    for output in app_langgraph.stream(inputs):
        for key, value in output.items():
            print(f"\n--- [Étape : {key}] ---")
            if "sql_query" in value:
                print(f"SQL : {value['sql_query']}")
            if "error" in value and value["error"]:
                print(f" Erreur : {value['error']}")
            if "final_response" in value:
                print(f"\n RÉPONSE FINALE :\n{value['final_response']}")

if __name__ == "__main__":
    main()