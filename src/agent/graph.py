# agent/graph.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import Nodes
import logging

logger = logging.getLogger(__name__)

def create_graph(agent_instance, engines_dict): #: ReActAgent

    # 1. Initialisation avec notre structure d'état
    workflow = StateGraph(AgentState)
    
    # 2. Instanciation de ta classe de nœuds
    nodes = Nodes(agent=agent_instance,engines=engines_dict)
    
    # 3. Ajout des nœuds au schéma
    workflow.add_node("assistant", nodes.call_model)
    workflow.add_node("validateur", nodes.check_pandas_syntax)
    workflow.add_node("executeur", nodes.execute_tool)
    workflow.add_node("disputeur", nodes.critique_response)
    
    # 4. Configuration des chemins (Edges)
    workflow.set_entry_point("assistant")
    
    # Lien direct : après l'assistant, on vérifie TOUJOURS la syntaxe
    workflow.add_edge("assistant", "validateur")
    
    # Lien après validation
    workflow.add_conditional_edges(
        "validateur",
        lambda state: state["next_step"],
        {
            "execute": "executeur",
            "retry": "assistant"
        }
    )
    
    # Lien après exécution : toujours vers le critique
    workflow.add_edge("executeur", "disputeur")
    
    # Lien après critique
    workflow.add_conditional_edges(
        "disputeur",
        lambda state: state["next_step"],
        {
            "end": END,            # Critique acceptée, fin
            "retry": "assistant"   # Critique négative, on recommence
        }
    )
    
    logger.info("Graph créé avec succès (assistant -> validateur -> executeur -> disputeur -> END)")
    return workflow.compile()