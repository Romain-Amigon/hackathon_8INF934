# agent/nodes.py
import os
from llama_index.core.agent import ReActAgent
from state import AgentState
import asyncio

class Nodes:
    def __init__(self, agent, engines):
        self.agent = agent
        self.engines = engines # Un dictionnaire contenant tes moteurs

    async def call_model(self, state: AgentState):
        user_input = state["messages"][-1]
        if hasattr(user_input, 'content'):
            user_input = user_input.content
            
        # Logique de routage ultra-simple pour ton test :
        # Si on parle de nids-de-poule, on utilise le moteur 311 directement
        if "Nid-de-poule" in user_input or "311" in user_input:
            response = self.engines["311"].query(user_input)
            return {
                "messages": [str(response)], 
                "next_step": "end" 
            }
        
        # Sinon, on utilise l'agent ReAct (qui fait les 429)
        # Mais on ajoute un sleep pour protéger l'API
        await asyncio.sleep(2)
        response = await self.agent.achat(user_input)
        return {
            "messages": [response.response], 
            "next_step": "check_syntax" 
        }
    
    def check_pandas_syntax(self, state: AgentState):
        """Vérifie le format du code avant exécution."""
        last_msg = str(state["messages"][-1].content)
        
        # Détection des erreurs de format
        if "df =" in last_msg or "```" in last_msg:
            return {
                "messages": ["ERREUR : Format invalide. Ne pas utiliser 'df =' ni de markdown."],
                "next_step": "retry"
            }
        
        return {"next_step": "execute"}
    

    def execute_tool(self, state: AgentState):
        """Validation finale du résultat."""
        last_msg = str(state["messages"][-1].content)
        
        if "Error" in last_msg:
            return {"next_step": "retry"}
            
        return {"next_step": "end"}