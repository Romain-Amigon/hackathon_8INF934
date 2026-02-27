# agent/nodes.py
import os
from llama_index.core.agent import ReActAgent
from src.agent.state import AgentState
import asyncio
import numpy as np
import pandas as pd
import time
import logging


logger = logging.getLogger(__name__)

class Nodes:
    def __init__(self, agent, engines,retriever):
        self.agent = agent
        self.engines = engines # Un dictionnaire contenant tes moteurs
        self.retriever = retriever

    async def routeur_initial(self, state: AgentState):
        question = state["messages"][0].content if hasattr(state["messages"][0], 'content') else str(state["messages"][0])
        
        prompt = f"""Tu es un routeur logique pour un assistant de la Ville de Montréal.
        Question de l'utilisateur : {question}
        
        Doit-on interroger une base de données de statistiques pour répondre ?
        Les sujets de tes bases de données incluent : 
        - Les requêtes 311 (nids-de-poule, déneigement, etc.)
        - Les collisions routières (accidents, morts, blessés, etc.)
        - La météo (neige, pluie, température, etc.)
        
        Si la question demande de compter ("combien"), de faire une moyenne, de comparer des dates, ou d'obtenir un chiffre précis sur ces sujets, réponds "OUI".
        Si la question demande une définition, un règlement ou une explication générale (ex: "qu'est-ce qu'un accident grave ?"), réponds "NON".
        Si la question ne concerne pas Montreal de façon explicite, donc une autre ville est nommé, réponds "NON".
        
        Réponds UNIQUEMENT par 'OUI' ou 'NON'."""
        
        reponse = await self.agent.llm.acomplete(prompt)
        texte = reponse.text.strip().upper()
        
        trace = [f"--- ROUTEUR PROMPT ---\n{prompt}", f"--- ROUTEUR RÉPONSE ---\n{reponse.text}"]
        
        if "OUI" in texte:
            return {"next_step": "pandas_avec_rag", "reflexions": trace}
        else:
            return {"next_step": "rag_seul", "reflexions": trace}
      

    async def recherche_lexique(self, state: AgentState):
        time.sleep(0.1)
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
            Si le dernier message est 'Je n'ai pas réussi à extraire cette donnée après plusieurs tentatives. Pouvez-vous reformuler la question ?' précise bien dans ta réponse qu'il y a eu une erreur.            
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
        time.sleep(0.1)
        iterations = state.get("iteration_count", 0)
        
        if iterations >= 2:
            message_echec = "Je n'ai pas réussi à extraire cette donnée après plusieurs tentatives. Pouvez-vous reformuler la question ?"
            trace = ["--- ARRÊT FORCÉ ---\nLimite d'itérations atteinte pour la génération de code."]
            return {
                "messages": [message_echec],
                "next_step": "generation",
                "iteration_count": iterations + 1,
                "reflexions": trace
            }
        historique = state["messages"]
        question_initiale = historique[0].content if hasattr(historique[0], 'content') else str(historique[0])
        
        logger.info(f"🔵 ASSISTANT: Traitement de la question: {question_initiale[:100]}...")
        
        # ON RÉCUPÈRE DYNAMIQUEMENT TES DESCRIPTIONS
        desc_311 = self.agent.tools[0].metadata.description
        desc_coll = self.agent.tools[1].metadata.description
        desc_meteo = self.agent.tools[2].metadata.description
        desc_metro = self.agent.tools[3].metadata.description
    
        dernier_feedback = ""
        if len(historique) > 1:
            dernier_feedback = f"\nATTENTION : Ton essai précédent a échoué. Erreur : {historique[-1].content}. Ne refais pas la même erreur."
            logger.warning(f"⚠️  RETRY MODE: {dernier_feedback[:80]}...")
    
        prompt = f"""Tu es un analyste de données expert pour la ville de Montréal.
        Réponds UNIQUEMENT avec des lignes de code Python.
        
        CONSIGNES STRICTES ET OBLIGATOIRES :
        - Syntaxe OBLIGATOIRE pour les dates : df_nom['nom_col'] = pd.to_datetime(df_nom['nom_col'], format='mixed', errors='coerce')
        - AVERTISSEMENT FATAL : Applique pd.to_datetime() UNIQUEMENT sur les colonnes 'DATE', 'DDS_DATE_CREATION' ou 'DATE_DERNIER_STATUT'.
        - NE CONVERTIS JAMAIS les colonnes géographiques (LOC_LAT, LOC_LONG, stop_name) en date.
        - Repond avec une variable nommée resultat contenant une valeur simple.
        
        CONSIGNES SPATIALES ET DE DISTANCE (ÉVITER LES CRASHS MÉMOIRE) :
        - SI ET SEULEMENT SI la question parle explicitement de distance ou de proximité (ex: "à moins de 200m"), tu DOIS filtrer ton DataFrame principal puis utiliser filtrer_proches(df_points, df_cibles, rayon).
        - SI tu utilises filtrer_proches ne met pas de notes pour le dire, donne juste le code python.
        - SI LA QUESTION NE PARLE PAS DE DISTANCE, N'UTILISE SURTOUT PAS filtrer_proches ni df_metro. Fais un simple comptage ou filtre Pandas.
        - Exemple  : 
            df_coll['DATE'] = pd.to_datetime(df_coll['DATE'], format='mixed', errors='coerce')
            df_filtre = df_coll[(df_coll['DATE'].dt.year == 2021) & (df_coll['GRAVITE'] == 'Grave')]
            df_proches = filtrer_proches(df_filtre, df_metro, 200)
            resultat = len(df_proches)
        
        VOICI TES BASES DE DONNÉES (DÉJÀ CHARGÉES) :
        1. 'df_311' : {desc_311}
        2. 'df_coll' : {desc_coll}
        3. 'df_meteo' : {desc_meteo}
        4. 'df_metro' : {desc_metro}
        
        QUESTION : {question_initiale}
        {dernier_feedback}
        
        Code Python :"""
    
        response = await self.agent.llm.acomplete(prompt)
        
        #response=response.replace("```python",'').replace("```","")
        trace = [f" --- derniere erreur :  {dernier_feedback}\n",f"--- ANALYSTE RÉPONSE CODE PANDAS ---\n{response.text}"]
        logger.info(f"✅ CODE GÉNÉRÉ: {response.text.strip()[:150]}...")
            
        return {
            "messages": [response.text.strip()], 
            "next_step": "execute",
            "iteration_count": iterations + 1,
            "reflexions": trace
        }

    def check_pandas_syntax(self, state: AgentState):
        """Vérifie le format du code avant exécution."""

        last_msg = str(state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else state["messages"][-1])
        trace = ["--- ANALYSTE SYNTAXE ---\n{NOT OK}"]
        logger.info("🔍 VALIDATEUR: Vérification de la syntaxe...")
        
        # Détection des erreurs de format
        if "df =" in last_msg :
            error_msg = "ERREUR : Format invalide. Ne pas utiliser 'df ='"
            logger.warning(f"❌ SYNTAXE: {error_msg}")
            return {

                "messages": ["ERREUR : Format invalide. Ne pas utiliser 'df ='"],
                "next_step": "retry",
                "reflexions": trace

            }
        """
        if  "```" in last_msg:
            error_msg = "ERREUR : Format invalide. Ne pas utiliser ``` ni de markdown."
            logger.warning(f"❌ SYNTAXE: {error_msg}")
            return {
                "messages": ["ERREUR : Format invalide. Ne pas utiliser ``` ni de markdown."],
                "next_step": "retry",
                "reflexions": trace

            }
        """
        if  'resultat' not in last_msg:
            error_msg = "ERREUR : Il est nécessaire d'enregistrer le résultat dans une variable nommée resultat"
            logger.warning(f"❌ SYNTAXE: {error_msg}")
            return {
                "messages": ["ERREUR : Il est nécessaire d'enregistrer le résultat dans une variable nommée resultat"],
                "next_step": "retry",
                "reflexions": trace
            }
        
        trace = ["--- ANALYSTE SYNTAXE ---\n{OK}"]
        logger.info("✅ VALIDATEUR: Syntaxe valide, passage à l'exécution")

        return {
            "next_step": "execute",
            "reflexions": trace
        }

    


    def execute_tool(self, state: AgentState):
        code_brut = str(state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else state["messages"][-1])
        clean_code = code_brut.replace("```python", "").replace("```", "").strip()
        
        
        def filtrer_proches(df_points, df_cibles, rayon):
            lat1 = np.radians(df_points['LOC_LAT'].values)[:, np.newaxis]
            lon1 = np.radians(df_points['LOC_LONG'].values)[:, np.newaxis]
            lat2 = np.radians(df_cibles['LOC_LAT'].values)
            lon2 = np.radians(df_cibles['LOC_LONG'].values)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distances = 6371000 * c
            
            proximite = distances <= rayon
            return df_points[proximite.any(axis=1)]
        logger.info(f"⚙️  EXECUTEUR: Exécution du code...")
        
        contexte_data = {
            "df_311": self.engines["311"]._df,
            "df_coll": self.engines["coll"]._df,
            "df_meteo": self.engines["meteo"]._df,
            "df_metro": self.engines["metro"]._df, 
            "filtrer_proches": filtrer_proches,
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
            
            
            trace = [f"--- ANALYSTE RÉPONSE ---\n Valeur trouvée : {final_val}"]
            logger.info(f"✅ EXECUTEUR: Résultat obtenu = {final_val}")



            return {
                "messages": [f"Le résultat de l'analyse est : {final_val}"],
                "next_step": "generation",
                "reflexions": trace

            }
        except Exception as e:
            erreur_courte = str(e)[:300]
            return {
                "messages": [f"ERREUR D'EXÉCUTION : {erreur_courte}"],
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