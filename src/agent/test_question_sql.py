import asyncio
import os
import torch
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import create_engine


device = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "cpu"
print(f"--- Info : Utilisation du device [{device}] pour l'embedding ---")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device=device
)


Settings.llm = Ollama(
    model="qwen2.5-coder:7b", 
    base_url="http://127.0.0.1:11434", 
    request_timeout=600.0
)

# 2. CONNEXION SQL
# Le fichier .db est monté via le volume Docker dans /app/mobility.db
db_path = "/app/data/raw/mobility.db"

if not os.path.exists(db_path):
    print(f" Erreur : {db_path} introuvable. Vérifie le montage du volume dans docker-compose.")
    # On peut créer une base vide ou sortir
    import sys; sys.exit(1)

engine = create_engine(f"sqlite:///{db_path}")
sql_database = SQLDatabase(engine)

# 3. MOTEUR DE REQUÊTE
# Note : LlamaIndex va lire les schémas de ces tables automatiquement
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["requetes_311", "collisions", "weather_montreal"]
)

def main():
    question = "Dans la table requetes_311, quel est le top 3 des ACTI_NOM les plus fréquents ?"
    
    try:
        print(f"\n--- ENVOI DE LA QUESTION À OLLAMA ---\n{question}")
        reponse = query_engine.query(question)
        
        print(f"\n--- RÉPONSE FINALE ---\n{reponse}")
        
        if hasattr(reponse, "metadata") and reponse.metadata.get('sql_query'):
            print(f"\n--- SQL GÉNÉRÉ DANS DOCKER ---\n{reponse.metadata.get('sql_query')}")
            
    except Exception as e:
        print(f" Erreur lors de l'exécution : {e}")

if __name__ == "__main__":
    main()