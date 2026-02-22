# hackathon_8INF934
---
## DATA




Transports en commun : https://www.stm.info/fr/a-propos/developpeurs/description-des-donnees-disponibles?utm_source=chatgpt.com
accidents : https://www.donneesquebec.ca/recherche/dataset/vmtl-collisions-routieres

Météo

Fréquentations (google maps ??)


date des data
```plaintext
collisions : 2012 a 2021

311: 2016 a 2026  (récupérer en local)

weather : 2012 a 2026
```


penser a citer la source des data
```plaintext
Attribution de la source à la Ville de Montréal
Lorsque vous réutilisez nos données et contenus, vous devez respecter les conditions suivantes :

Vous devez créditer les données et les contenus que vous utilisez et préciser si des modifications ont été effectuées ou si des interprétations en ont été tirées.

Vous ne pouvez pas indiquer ou suggérer que la Ville de Montréal vous soutient ou endosse votre usage de ses données et ses contenus.

Cette condition s’applique également à l’intégration des données de la Ville de Montréal à une base de données dont vous ou votre organisation êtes propriétaire.

Vous ne pouvez pas restreindre l’accès aux données et contenus originaux partagés par la Ville de Montréal, que ce soit soit sous la forme de conditions légales ou de mesures techniques.
Notre licence s'inspire et est conforme aux principes et à la définition du Savoir Libre de l’Open Knowledge Foundation. Elle a été adoptée par les instances de la Ville de Montréal le 27 février 2014 par la résolution CG14 0091 du Conseil d'agglomération. Ce choix vise à faciliter la réutilisation des données et à encourager les utilisateurs et utilisatrices des données ouvertes à redonner les bénéfices qu’ils en retirent à la communauté.

Comment citer les données ouvertes
Nous vous suggérons quelques façons de citer les données et les contenus mis à disposition par la Ville de Montréal. Nous vous invitons à les adapter à vos besoins et aux particularités de votre projet.

Rapports et documents académiques
Ville de Montréal. (Date de la dernière mise à jour de la ressource). Nom de l’ensemble de données [Ensemble de données]. Données ouvertes de la Ville de Montréal. Lien vers l’ensemble de données
Ville de Montréal. (2023). Indicateurs de condition des chaussées du réseau routier [Ensemble de données]. Données ouvertes de la Ville de Montréal. https://donnees.montreal.ca/dataset/condition-chaussees-reseau-routier
Cet exemple est structuré à titre indicatif selon la norme APA 7e génération. D’autres normes bibliographiques, comme l’EEE, pourraient également être employées.

Tout autre usage (incluant les solutions numériques)
Avec les données Nom de l’ensemble de données tirées du site web des données ouvertes de la Ville de Montréal en date du Date de consultation.
Avec les données Indicateurs de condition des chaussées du réseau routier tirées du site web des données ouvertes de la Ville de Montréal en date du 30 mars 2023.
```


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
