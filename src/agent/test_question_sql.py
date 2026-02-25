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
    # Exemple de question complexe pour tester ton moteur
    question = "Quels sont les 3 types de requêtes 311 les plus fréquents et y a-t-il une corrélation visuelle avec le top des collisions ?"
    
    try:
        print(f"\n[1/3]  RÉFLEXION : {question}")
        
        # LlamaIndex génère le SQL, l'exécute ET génère la réponse ici :
        reponse_objet = query_engine.query(question)
        
        # --- CRITÈRE 2 : VALIDATEUR & AFFICHAGE DU RÉSULTAT ---
        print("\n[2/3]  EXÉCUTION SQL RÉUSSIE")
        if hasattr(reponse_objet, "metadata") and reponse_objet.metadata.get('sql_query'):
            print(f"     Requête exécutée : \n     {reponse_objet.metadata.get('sql_query')}")
        
        print(f"\n--- RÉPONSE DE L'AGENT ---\n{reponse_objet.response}")

        # --- CRITÈRE 4 : MODE CONTRADICTEUR  ---
        print("\n[3/3]  ANALYSE CRITIQUE (Contradicteur)")
        
        prompt_critique = f"""
        En tant qu'expert en mobilité urbaine, analyse de façon critique cette réponse : '{reponse_objet.response}'
        Identifie :
        1. Une limite liée aux données (ex: saisonnalité, données s'arrêtant en 2021).
        2. Un risque d'interprétation.
        Sois bref et percutant.
        """
        # On réutilise Ollama pour la critique
        critique = Settings.llm.complete(prompt_critique)
        print(f"{critique}")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")

main()