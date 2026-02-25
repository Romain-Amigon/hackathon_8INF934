# agent/nodes.py
import os
from llama_index.core.agent import ReActAgent
from .state import AgentState
import asyncio
import logging

import pandas as pd

logger = logging.getLogger(__name__)

class Nodes:
    def __init__(self, agent, engines):
        self.agent = agent
        self.engines = engines # Un dictionnaire contenant tes moteurs
        self.llm_semaphore = asyncio.Semaphore(1)

    
    async def call_model(self, state: AgentState):
        historique = state["messages"]
        question_initiale = historique[0].content if hasattr(historique[0], 'content') else str(historique[0])
        
        logger.info(f"🔵 ASSISTANT: Traitement de la question: {question_initiale[:100]}...")
        
        # ON RÉCUPÈRE DYNAMIQUEMENT TES DESCRIPTIONS
        desc_311 = self.agent.tools[0].metadata.description
        desc_coll = self.agent.tools[1].metadata.description
        desc_meteo = self.agent.tools[2].metadata.description
    
        dernier_feedback = ""
        if len(historique) > 1:
            dernier_feedback = f"\nATTENTION : Ton essai précédent a échoué. Erreur : {historique[-1].content}. Ne refais pas la même erreur."
            logger.warning(f"⚠️  RETRY MODE: {dernier_feedback[:80]}...")
    
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
            logger.info(f"✅ CODE GÉNÉRÉ: {response.text.strip()[:150]}...")
            print(response)
            
        return {
            "messages": [response.text.strip()], 
            "next_step": "execute" 
        }

    def check_pandas_syntax(self, state: AgentState):
        """Vérifie le format du code avant exécution."""
        last_msg = str(state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else state["messages"][-1])
        
        logger.info("🔍 VALIDATEUR: Vérification de la syntaxe...")
        
        # Détection des erreurs de format
        if "df =" in last_msg :
            error_msg = "ERREUR : Format invalide. Ne pas utiliser 'df ='"
            logger.warning(f"❌ SYNTAXE: {error_msg}")
            return {
                "messages": [error_msg],
                "next_step": "retry"
            }
        if  "```" in last_msg:
            error_msg = "ERREUR : Format invalide. Ne pas utiliser ``` ni de markdown."
            logger.warning(f"❌ SYNTAXE: {error_msg}")
            return {
                "messages": [error_msg],
                "next_step": "retry"
            }
    
        if  'resultat' not in last_msg:
            error_msg = "ERREUR : Il est nécessaire d'enregistrer le résultat dans une variable nommée resultat"
            logger.warning(f"❌ SYNTAXE: {error_msg}")
            return {
                "messages": [error_msg],
                "next_step": "retry"
            }
        
        logger.info("✅ VALIDATEUR: Syntaxe valide, passage à l'exécution")
        return {
            "messages": [],  # Pas de new message, on garde l'historique
            "next_step": "execute"
        }
    


    def execute_tool(self, state: AgentState):
        code_brut = str(state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else state["messages"][-1])
        clean_code = code_brut.replace("```python", "").replace("```", "").strip()
        
        logger.info(f"⚙️  EXECUTEUR: Exécution du code...")
        
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
            
            logger.info(f"✅ EXECUTEUR: Résultat obtenu = {final_val}")
    
            return {
                "messages": [f"RÉSULTAT: {final_val}"],
                "next_step": "critique"
            }
        except Exception as e:
            logger.error(f"❌ EXECUTEUR: Erreur d'exécution: {str(e)}")
            return {
                "messages": [f"ERREUR D'EXÉCUTION : {str(e)}"],
                "next_step": "retry"
            }
    
    def critique_response(self, state: AgentState):
        """Nœud critique qui évalue la qualité de la réponse (Mode Contradicteur).
        Version synchrone avec support de l'asynchrone."""
        try:
            historique = state["messages"]
            
            # Extraction safe de la question initiale
            first_msg = historique[0]
            if hasattr(first_msg, 'content'):
                question_initiale = first_msg.content
            else:
                question_initiale = str(first_msg)
            
            # Extraction safe du dernier message (la réponse)
            last_msg = historique[-1]
            if hasattr(last_msg, 'content'):
                derniere_reponse = last_msg.content
            else:
                derniere_reponse = str(last_msg)
            
            logger.info(f"🎯 DISPUTEUR: Évaluation critique de: {derniere_reponse[:80]}...")
            
            # Prompt critique
            prompt_critique = f"""Tu es un critique analytique strict des analyses de données. Évalue cette réponse sur :
1. Exactitude factuelle (basée sur les données)
2. Complétude (répond-elle à toute la question ?)
3. Clarté et objectivité
4. Format numérique approprié

QUESTION: {question_initiale}
RÉPONSE: {derniere_reponse}

VERDICT (une seule ligne):
- ✅ si la réponse est VALIDE, COMPLÈTE et FACTUELLE
- ❌ si la réponse est INCOMPLÈTE, INCORRECTE ou VAGUE"""
            
            logger.info(f"📝 DISPUTEUR: Envoi de la critique au LLM...")
            
            # Obtenir ou créer une boucle d'événements
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Créer une coroutine pour la critique
            async def get_critique():
                async with self.llm_semaphore:
                    critique = await self.agent.llm.acomplete(prompt_critique)
                    return critique.text.strip() if hasattr(critique, 'text') else str(critique).strip()
            
            # Exécuter la coroutine
            critique_text = loop.run_until_complete(get_critique())
            
            logger.info(f"📋 DISPUTEUR: Critique reçue = {critique_text[:150]}")
            
            # Décision basée sur la critique
            if critique_text.startswith("✅"):
                logger.info("✅✅✅ DISPUTEUR: RÉPONSE VALIDÉE, FIN")
                return {
                    "messages": [f"\n═════════════════════════════\n✅ RÉPONSE ACCEPTÉE\n═════════════════════════════\n{derniere_reponse}\n\n🗣️ Critique: {critique_text}"],
                    "next_step": "end"
                }
            else:
                logger.warning(f"⚠️  DISPUTEUR: RÉPONSE REJETÉE - {critique_text[:80]}")
                return {
                    "messages": [f"🔄 RÉVISION REQUISE:\n{critique_text}\n\n(Le système va regénérer une meilleure réponse...)"],
                    "next_step": "retry"
                }
        except Exception as e:
            logger.error(f"💥 DISPUTEUR CRASH: {str(e)}", exc_info=True)
            return {
                "messages": [f"❌ Erreur critique: {str(e)}"],
                "next_step": "end"
            }