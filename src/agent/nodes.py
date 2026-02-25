# agent/nodes.py
import os
from llama_index.core.agent import ReActAgent
from state import AgentState
import asyncio

import pandas as pd

class Nodes:
    def __init__(self, agent, engines,retriever):
        self.agent = agent
        self.engines = engines # Un dictionnaire contenant tes moteurs
        self.retriever = retriever

    async def routeur_initial(self, state: AgentState):
        question = state["messages"][0].content if hasattr(state["messages"][0], 'content') else str(state["messages"][0])
        
        prompt = f"""Tu es un routeur.
        Question: {question}
        La question demande-t-elle un calcul, un comptage ou une statistique sur des bases de données de mobilité ?
        Réponds uniquement par 'OUI' ou 'NON'."""
        
        reponse = await self.agent.llm.acomplete(prompt)

        texte = reponse.text.strip().upper()
        
        trace = [f"--- ROUTEUR PROMPT ---\n{prompt}", f"--- ROUTEUR RÉPONSE ---\n{reponse.text}"]
        
        if "OUI" in texte:
            return {"next_step": "pandas_avec_rag", "reflexions": trace}
        else:
            return {"next_step": "rag_seul", "reflexions": trace}
      

    async def recherche_lexique(self, state: AgentState):
        question = state["messages"][0].content if hasattr(state["messages"][0], 'content') else str(state["messages"][0])
        etape_precedente = state.get("next_step", "pandas_avec_rag")
        
        docs = self.retriever.retrieve(question)
        contexte = "\n".join([doc.text for doc in docs])
        
        if etape_precedente == "rag_seul":
            prompt = f"""Réponds à la question en utilisant le contexte fourni, synthétise la réponse.
            Contexte: {contexte}
            Question: {question}"""
            
            reponse = await self.agent.llm.acomplete(prompt)
                
            return {"messages": [reponse.text], "next_step": "end"}
    
        elif etape_precedente == "generation":
            stat = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])
            
            prompt = f"""Tu es un expert en mobilité urbaine à Montréal.
            Tu viens de calculer avec précision la donnée suivante à partir des bases de données de la ville : {stat}
            
            Ce chiffre est la vérité absolue et constitue la réponse directe à la question. Ne cherche pas à le vérifier dans le glossaire.
            
            Question de l'utilisateur : {question}
            
            Contexte issu du glossaire (à utiliser UNIQUEMENT pour enrichir les définitions ou expliquer le phénomène) : 
            {contexte}
            
            Rédige une synthèse fluide en langage naturel. Intègre la statistique ({stat}) et utilise le contexte pour donner du sens à ce chiffre."""
            
            reponse = await self.agent.llm.acomplete(prompt)
                
            return {"messages": [reponse.text], "next_step": "end"}
            
        else:
            message_contexte = f"INFO GLOSSAIRE POUR PANDAS: {contexte}"
            return {"messages": [message_contexte], "next_step": "assistant"}
        
    async def call_model(self, state: AgentState):
        historique = state["messages"]
        question_initiale = historique[0].content if hasattr(historique[0], 'content') else str(historique[0])
        
        # ON RÉCUPÈRE DYNAMIQUEMENT TES DESCRIPTIONS
        desc_311 = self.agent.tools[0].metadata.description
        desc_coll = self.agent.tools[1].metadata.description
        desc_meteo = self.agent.tools[2].metadata.description
    
        dernier_feedback = ""
        if len(historique) > 1:
            dernier_feedback = f"\nATTENTION : Ton essai précédent a échoué. Erreur : {historique[-1].content}. Ne refais pas la même erreur."
    
        prompt = f"""Tu es un analyste de données expert pour la ville de Montréal.
        Réponds UNIQUEMENT avec des lignes de code Python, ne fais pas de commentaires et n'utilise pas ```
        
        CONSIGNES DE JOINTURE :
            - Si la question nécessite de croiser deux bases (ex: météo et accidents), fais une jointure (merge) sur les colonnes de dates.
            - Assure-toi que les deux colonnes de dates sont au format datetime avant le merge.
            - Exemple de logique : 
                1. Filtrer df_meteo pour garder les jours > 10cm.
                2. Joindre ce résultat avec df_coll sur la date.
                3. Compter les lignes du résultat final et stocker le résultat dans une variable resultat.
        
        VOICI TES BASES DE DONNÉES (DÉJÀ CHARGÉES) :
        1. 'df_311' : {desc_311}
        2. 'df_coll' : {desc_coll}
        3. 'df_meteo' : {desc_meteo}
    
        QUESTION : {question_initiale}
        {dernier_feedback}
    
        CONSIGNES :
        - Réponds UNIQUEMENT avec des lignes de code Python utilisant les noms df_311, df_coll ou df_meteo.
        - Pas de markdown (```), pas de 'import pandas', pas de création de données fictives.
        - Pour les dates, utilise pd.to_datetime().
        -repond avec une variable nommée resultat
        
        Code Python :"""
    
        response = await self.agent.llm.acomplete(prompt)
        trace = [f"--- ANALYSTE PROMPT ---\n{prompt}", f"--- ANALYSTE RÉPONSE ---\n{response.text}"]
            
        return {
            "messages": [response.text.strip()], 
            "next_step": "execute",
            "reflexions": trace
        }

    def check_pandas_syntax(self, state: AgentState):
        """Vérifie le format du code avant exécution."""
        last_msg = str(state["messages"][-1].content)
        trace = ["--- ANALYSTE SYNTAXE ---\n{NOT OK}"]
        # Détection des erreurs de format
        if "df =" in last_msg :
            return {
                "messages": ["ERREUR : Format invalide. Ne pas utiliser 'df ='"],
                "next_step": "retry",
                "reflexions": trace
            }
        if  "```" in last_msg:
            return {
                "messages": ["ERREUR : Format invalide. Ne pas utiliser ``` ni de markdown."],
                "next_step": "retry",
                "reflexions": trace
            }
    
        if  'resultat' not in last_msg:
            return {
                "messages": ["ERREUR : Il est nécessaire d'enregistrer le résultat dans une variable nommée resultat"],
                "next_step": "retry",
                "reflexions": trace
            }
        
        trace = ["--- ANALYSTE SYNTAXE ---\n{OK}"]
            
        return {
            "next_step": "execute",
            "reflexions": trace
        }

    


    def execute_tool(self, state: AgentState):
        code_brut = str(state["messages"][-1].content)
        clean_code = code_brut.replace("```python", "").replace("```", "").strip()
        
        contexte_data = {
            "df_311": self.engines["311"]._df,
            "df_coll": self.engines["coll"]._df,
            "df_meteo": self.engines["meteo"]._df,
            "pd": pd,
            "resultat": None  # On prépare une variable pour stocker la réponse
        }
    
        try:
            # On demande au modèle d'assigner sa réponse finale à la variable 'resultat'
            # ou on tente d'exécuter le bloc de code
            exec(clean_code, {"__builtins__": __builtins__}, contexte_data)
            
            # Si le modèle a créé une variable 'diff_accidents' ou 'resultat'
            # On essaie de récupérer une valeur logique
            final_val = contexte_data.get("resultat") or contexte_data.get("diff_accidents") or "Calcul effectué sans valeur de retour spécifique"
    
            return {
                "messages": [f"Le résultat de l'analyse est : {final_val}"],
                "next_step": "generation"
            }
        except Exception as e:
            return {
                "messages": [f"ERREUR D'EXÉCUTION : {str(e)}"],
                "next_step": "retry"
            }