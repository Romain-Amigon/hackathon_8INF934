# hackathon_8INF934
---
## DATA


### Stats


Transports en commun : https://www.stm.info/fr/a-propos/developpeurs/description-des-donnees-disponibles?utm_source=chatgpt.com
accidents : https://www.donneesquebec.ca/recherche/dataset/vmtl-collisions-routieres

Météo

Fréquentations (maps ??)

### NLP
requetes générales : https://donnees.montreal.ca/dataset/requete-311

---
## Technologies
LangGraph (Orchestration) : C'est le choix idéal pour implémenter les cycles de "réflexion" et de "contradiction" demandés. Contrairement à une chaîne RAG linéaire, LangGraph permet de créer des boucles de rétroaction où un agent peut valider une requête SQL avant de l'exécuter.

LlamaIndex (Gestion des données) : Utilisez-le pour charger vos fichiers (CSV des collisions, 311) et les transformer en "Query Engines" prêts à l'emploi. Il est 40% plus rapide que les méthodes classiques pour la récupération de documents.

OLlama pour un llm local pour RGPD et éviter clé payante

Modèle,             Spécialité,                         Recommandation pour ce projet
qwen2.5-coder:7b,   Écriture de code (Python/SQL),      Idéal pour le nœud de génération Pandas.
llama3.1:8b,        Raisonnement général et RAG,        "Très bon pour l'orchestration LangGraph, la détection d'ambiguïté et le contradicteur."
mistral:7b,                                             Suivi d'instructions,Bonne alternative de secours si Llama est trop lent sur votre machine.
---
## Architecture 

Prototype

mobility-copilot/
│
├── data/
│   ├── raw/                  
│   └── vector_store/         
│
├── src/
│   ├── __init__.py
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app.py            
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py          
│   │   └── nodes.py          
│   └── data_pipeline/
│       ├── __init__.py
│       └── ingestion.py      
│
├── .env                      
├── .gitignore                
├── requirements.txt          
└── README.md

---
## Vérification

Faire un dataset de vérif {question : réponse attendue} et comparer la réponse attendue par rapport à celle recue en cos_similarity dans l'espace emb
