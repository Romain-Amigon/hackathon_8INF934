import streamlit as st
import pandas as pd
import torch
import os
import sqlalchemy
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import create_engine, inspect

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Mobility Copilot - Hackathon 2026", layout="wide")
st.title(" Mobility Copilot : Analyse Intelligente")

# --- INITIALISATION LLM ---
@st.cache_resource
def load_settings():
    device = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "cpu"
    # Utilisation de modèles de reconnaissance/embedding pré-entraînés [cite: 61]
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device=device)
    llm = Ollama(model="qwen2.5-coder:7b", base_url="http://127.0.0.1:11434", request_timeout=600.0)
    Settings.embed_model = embed_model
    Settings.llm = llm
    return llm

llm = load_settings()

# --- CONNEXION & EXTRACTION DU SCHÉMA (CdC: Validation & Précision) ---
db_path = "/app/data/raw/mobility.db"
engine = create_engine(f"sqlite:///{db_path}")
sql_database = SQLDatabase(engine)

def get_real_schema():
    """Extrait les vrais noms de colonnes pour éviter les hallucinations SQL"""
    inst = inspect(engine)
    schema_info = ""
    for table in ["requetes_311", "collisions", "weather_montreal"]:
        cols = [c['name'] for c in inst.get_columns(table)]
        schema_info += f"- Table '{table}': Colonnes {cols}\n"
    return schema_info

# --- MOTEUR DE REQUÊTE ---
descriptions_tables = {
    "requetes_311": "Signalements. Clé de jointure temporelle: 'DDS_DATE_CREATION'.",
    "collisions": "Accidents. Clé de jointure temporelle: 'DT_ACCDN'. Colonne gravité: 'GRAVITE'.",
    "weather_montreal": "Météo. Clé de jointure: 'time'. Colonnes: 'precipitation_sum', 'snowfall_sum'."
}

# Dans ta boucle for attempt in range(max_retries):
# Remplace le current_query par ce prompt plus direct :


query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["requetes_311", "collisions", "weather_montreal"],
    context_query_kwargs=descriptions_tables
)

# --- INTERFACE ---
st.sidebar.header(" Schéma de la Base")
with st.sidebar.expander("Voir les colonnes réelles"):
    st.text(get_real_schema())

question = st.text_input("Posez votre question :")

# --- BOUCLE DE RÉCURSION / RÉFLEXION ---
if st.button("Analyser"):
    if question:
        max_retries = 3
        current_query = question
        
        for attempt in range(max_retries):
            with st.spinner(f"Tentative {attempt+1}/{max_retries}..."):
                try:
                    # L'outil génère et valide la transcription SQL [cite: 54, 63]
                    response_obj = query_engine.query(current_query)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.success(" Résultat trouvé")
                        st.write(response_obj.response)
                        with st.expander("Requête SQL utilisée"):
                            st.code(response_obj.metadata.get('sql_query'), language='sql')
                    
                    with col2:
                        # Critère 4: Mode Contradicteur / Évaluation des performances [cite: 69]
                        st.subheader(" Critique")
                        prompt_critique = f"Analyse cette réponse : '{response_obj.response}'. Sois critique sur la précision."
                        st.warning(llm.complete(prompt_critique).text)
                    break # Succès !

                except Exception as e:
                    error_msg = str(e)
                    if attempt < max_retries - 1:
                        st.error(f"Erreur SQL. L'agent tente de se corriger...")
                        # On réinjecte l'erreur dans le prompt pour la correction [cite: 62]
                        current_query = f"""
                        ERREUR PRÉCÉDENTE : {error_msg}
                        CONSIGNE DE CORRECTION : 
                        1. Ne mélange pas les colonnes : 'snowfall_sum' est UNIQUEMENT dans 'weather_montreal'.
                        2. Pour corréler avec la météo, tu DOIS faire un JOIN avec 'weather_montreal' sur la date.
                        3. STRUCTURE RÉELLE : {get_real_schema()}
                        QUESTION INITIALE : {question}
                        """
                    else:
                        st.error(f"Échec final après {max_retries} tentatives. Erreur: {error_msg}")
    else:
        st.warning("Entrez une question.")

st.caption("Hackathon IA 2026 - Prototype fonctionnel [cite: 30, 68]")