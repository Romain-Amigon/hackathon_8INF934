# agent/nodes.py
import os
from llama_index.core.agent import ReActAgent
from state import AgentState
import asyncio

import pandas as pd

class Nodes:
    def __init__(self, agent, engines):
        self.agent = agent
        self.engines = engines # Un dictionnaire contenant tes moteurs
        self.llm_semaphore = asyncio.Semaphore(1)


    
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
    
        async with self.llm_semaphore:
            response = await self.agent.llm.acomplete(prompt)
            print(response)
            
        return {
            "messages": [response.text.strip()], 
            "next_step": "execute" 
        }
        """
        user_input = state["messages"][-1]
        if hasattr(user_input, 'content'):
            user_input = user_input.content
                
        # Routage direct (Sans appel LLM coûteux)
        if any(keyword in user_input.lower() for keyword in ["nid-de-poule", "311"]):
            # Note: query() est souvent synchrone, pas besoin de semaphore ici
            response = self.engines["311"].query(user_input)
            return {"messages": [str(response)], "next_step": "end"}
    
        # Appel Agent avec protection stricte
        async with self.llm_semaphore:
            print("--- Entrée dans le Sémaphore (Verrouillé) ---")
            try:
                response = await self.agent.run(user_input)
                
                # PAUSE OBLIGATOIRE avant de libérer le verrou
                # Cela force Groq à respirer pendant 3 secondes avant la prochaine requête
                await asyncio.sleep(3) 
                
                return {
                    "messages": [response.response], 
                    "next_step": "check_syntax"
                }
            finally:
                print("--- Sortie du Sémaphore (Libéré) ---")
        """
    def check_pandas_syntax(self, state: AgentState):
        """Vérifie le format du code avant exécution."""
        last_msg = str(state["messages"][-1].content)
        
        # Détection des erreurs de format
        if "df =" in last_msg :
            return {
                "messages": ["ERREUR : Format invalide. Ne pas utiliser 'df ='"],
                "next_step": "retry"
            }
        if  "```" in last_msg:
            return {
                "messages": ["ERREUR : Format invalide. Ne pas utiliser ``` ni de markdown."],
                "next_step": "retry"
            }
    
        if  'resultat' not in last_msg:
            return {
                "messages": ["ERREUR : Il est nécessaire d'enregistrer le résultat dans une variable nommée resultat"],
                "next_step": "retry"
            }
        return {"next_step": "execute"}
    


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
                "next_step": "end"
            }
        except Exception as e:
            return {
                "messages": [f"ERREUR D'EXÉCUTION : {str(e)}"],
                "next_step": "retry"
            }