# hackathon_8INF934 - Mobility Copilot

## Quickstart

### Pré-requis

- Ollama installé et configuré [Download Ollama](https://ollama.com/download)
- Un environnement python avec les modules nécessaires au projet ``pip install -r requirements``
- OU pour une carte Nvidia utiliser le conteneur Docker ``docker compose up -d --build``

### Télécharger les données

```bash
python3 ./download_datasets.py
```

### Lancement de l'interface utilisateur

```bash
streamlit run ./src/ui/app.py
```

## Architecture du projet

Ah, je vois ! Ton visualiseur semble ignorer les sauts de ligne simples et tout "aplatir". C'est un problème classique si le texte n'est pas strictement encadré par des balises de bloc de code.

Pour forcer l'affichage en mode "arborescence" (monospace) dans n'importe quel lecteur Markdown, utilise exactement ce bloc ci-dessous :

```text
src/
├── agent/
│   ├── api_groq.py
│   ├── graph.py
│   ├── main.py
│   ├── nodes.py
│   ├── state.py
│   └── tests_poubelles/
│       ├── gemini.py
│       ├── ollama.py
│       └── test_question.py
├── data_pipeline/
│   ├── ingestion.py
│   └── transform.py
├── reports/
│   ├── briefing.py
│   ├── formatter.py
│   ├── hotspots.py
│   ├── trends.py
│   └── weak_signals.py
└── ui/
    ├── app.py
    ├── question_example.wav
    ├── README_Sentence.md
    ├── sentence.py
    └── testSentence.py

```

### 1. Vue d'ensemble
Notre application est un assistant analytique "data-grounded" qui croise trois sources de données majeures de la Ville de Montréal :
- **Requêtes 311** (Nids-de-poule, déneigement, etc.)
- **Collisions routières** (Localisation, gravité, victimes)
- **Données météorologiques** (Températures, précipitations, neige)

L'architecture repose sur un **Agent RAG (Retrieval-Augmented Generation)** supervisé par un graphe d'états.

### 2. Organisation des Fichiers (`src/`)

### `agent/` (Cœur de l'IA)
C'est ici que réside l'intelligence du système.
- **`state.py`** : Définit l'état partagé (`AgentState`) qui circule entre les nœuds (historique des messages, code généré, erreurs, etc.).
- **`graph.py`** : Définit le flux de travail avec **LangGraph**. Le cycle est : `assistant` ➔ `validateur` ➔ `executeur` ➔ `disputeur`.
- **`nodes.py`** : Contient la logique métier de chaque étape :
    - **Assistant** : Génère du code Python/Pandas via l'LLM (Groq/Llama 3.1).
    - **Validateur** : Vérifie que le code produit respecte les consignes de sécurité et de format.
    - **Exécuteur** : Lance réellement le code sur les DataFrames chargés en mémoire.
    - **Disputeur (Mode Contradicteur)** : Analyse de manière critique le résultat pour détecter d'éventuelles erreurs ou limites avant la réponse finale.
- **`api_groq.py` & `main.py`** : Points d'entrée pour l'initialisation de l'LLM, des moteurs de requête LlamaIndex (`PandasQueryEngine`) et des outils.

### `reports/` (Intelligence Analytique)
Modules spécialisés dans le traitement statistique lourd.
- **`hotspots.py`** : Identifie les zones critiques via un clustering spatial (**K-Means**) pour les collisions et des agrégations par arrondissement pour le 311.
- **`trends.py`** : Calcule les évolutions temporelles (YoY : Année sur Année, MoM : Mois sur Mois) et détecte les pics horaires.
- **`weak_signals.py`** : Détecte les **signaux faibles** via des régressions linéaires (croissance lente mais régulière) et des anomalies par **Z-score**.
- **`briefing.py`** : Agrège les résultats des modules précédents pour générer une synthèse hebdomadaire structurée.

### `data_pipeline/` (Gestion des données)
- **`ingestion.py`** : Chargement des fichiers CSV bruts.
- **`transform.py`** : Nettoyage, normalisation des types (dates, numérique) et préparation des jointures.

### `ui/` (Interface utilisateur)
- **`app.py`** : Application **Streamlit** offrant un tableau de bord interactif (Heatmaps, graphiques de tendances) et un chat avec l'agent.

### 3. Flux de Travail d'une Requête (Workflow)

1. **Input** : L'utilisateur pose une question (ex: "Impact de la neige sur les accidents").
2. **RAG / Planning** : L'agent consulte les descriptions des datasets (métadonnées) pour choisir les colonnes pertinentes.
3. **Génération de Code** : L'LLM produit un script Pandas effectuant la jointure et le filtrage.
4. **Validation** : Le système vérifie que le code n'est pas dangereux et qu'il stocke le résultat dans la variable attendue.
5. **Exécution** : Le code est exécuté sur les données réelles de Montréal.
6. **Contradiction** : Un deuxième appel LLM vérifie si le résultat semble cohérent ou s'il y a des risques d'interprétation.
7. **Output** : La réponse est affichée avec ses "preuves" (chiffres exacts).
