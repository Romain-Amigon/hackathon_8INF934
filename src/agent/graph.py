# agent/graph.py
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import Nodes

from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import Nodes

def create_graph(agent_instance, engines_dict, retriever=None):
    workflow = StateGraph(AgentState)
    
    nodes = Nodes(agent=agent_instance, engines=engines_dict, retriever=retriever)
    
    workflow.add_node("routeur", nodes.routeur_initial)
    workflow.add_node("rag_textuel", nodes.recherche_lexique)
    workflow.add_node("assistant", nodes.call_model)
    workflow.add_node("validateur", nodes.check_pandas_syntax)
    workflow.add_node("executeur", nodes.execute_tool)
    
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
    
    workflow.add_edge("assistant", "validateur")
    
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
    
    return workflow.compile()