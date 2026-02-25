# agent/graph.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import Nodes

def create_graph(agent_instance, engines_dict): #: ReActAgent

    # 1. Initialisation avec notre structure d'état
    workflow = StateGraph(AgentState)
    
    # 2. Instanciation de ta classe de nœuds
    nodes = Nodes(agent=agent_instance,engines=engines_dict)
    
    # 3. Ajout des nœuds au schéma
    workflow.add_node("assistant", nodes.call_model)
    workflow.add_node("validateur", nodes.check_pandas_syntax)
    workflow.add_node("executeur", nodes.execute_tool)
    
    # 4. Configuration des chemins (Edges)
    workflow.set_entry_point("assistant")
    
    # Lien direct : après l'assistant, on vérifie TOUJOURS la syntaxe
    workflow.add_edge("assistant", "validateur")
    
    # Lien conditionnel : que faire après la validation ?
    workflow.add_conditional_edges(
        "validateur",
        lambda state: state["next_step"], # On regarde la valeur dans le state
        {
            "execute": "executeur", # Tout est bon, on lance le calcul
            "retry": "assistant"    # Erreur détectée, on redemande au LLM de corriger
        }
    )
    
    # Lien après exécution
    workflow.add_conditional_edges(
        "executeur",
        lambda state: state["next_step"],
        {
            "end": END,            # Terminé, on rend la réponse
            "retry": "assistant"   # Le calcul a planté, on repart au début avec l'erreur
        }
    )
    
    return workflow.compile()