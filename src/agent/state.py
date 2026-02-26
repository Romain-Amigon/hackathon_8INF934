# agent/state.py
from typing import Annotated, TypedDict, List, Union
from langgraph.graph.message import add_messages



def add_reflexions(left: list, right: list) -> list:
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right
class AgentState(TypedDict):
    # add_messages permet d'accumuler l'historique automatiquement
    messages: Annotated[List[Union[dict, str]], add_messages]
    # On ajoute des champs pour suivre l'exécution
    current_code: str
    last_error: str
    iteration_count: int
    next_step: str
    
    reflexions: Annotated[List[str], add_reflexions]