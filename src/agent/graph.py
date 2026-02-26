# agent/graph.py
from langgraph.graph import StateGraph, END

import logging

logger = logging.getLogger(__name__)

from src.agent.state import AgentState
from src.agent.nodes import Nodes

def create_graph(agent_instance, engines_dict, retriever=None):
    workflow = StateGraph(AgentState)
    
    nodes = Nodes(agent=agent_instance, engines=engines_dict, retriever=retriever)
    
    workflow.add_node("routeur", nodes.routeur_initial)
    workflow.add_node("rag_textuel", nodes.recherche_lexique)
    workflow.add_node("assistant", nodes.call_model)
    workflow.add_node("validateur", nodes.check_pandas_syntax)
    workflow.add_node("executeur", nodes.execute_tool)
    workflow.add_node("disputeur", nodes.critique_response)
    
    workflow.set_entry_point("routeur")
    
    workflow.add_conditional_edges(
        "routeur",
        lambda state: state["next_step"],
        {
            "rag_seul": "rag_textuel",
            "pandas_avec_rag": "assistant"
        }
    )
    
    workflow.add_conditional_edges(
        "rag_textuel",
        lambda state: state["next_step"],
        {
            "assistant": "assistant",
            "end": END
        }
    )
    
    #workflow.add_edge("assistant", "validateur")
    workflow.add_conditional_edges(
        "assistant",
        lambda state: state["next_step"],
        {
            "execute": "validateur",
            "generation": "rag_textuel",
            "end": END
        }
    )    
    workflow.add_conditional_edges(
        "validateur",
        lambda state: state["next_step"],
        {
            "execute": "executeur",
            "retry": "assistant"
        }
    )
    
    workflow.add_conditional_edges(
        "executeur",
        lambda state: state["next_step"],
        {
            "generation":"rag_textuel",

            "retry": "assistant"
        }
    )
    """
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
    )"""
    
    logger.info("Graph créé avec succès (assistant -> validateur -> executeur -> generation augmentée -> END)")
    return workflow.compile()