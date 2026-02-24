import os
from typing import Literal
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# Configuration API Groq
api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)

# Configuration Base de données
db = SQLDatabase.from_uri("sqlite:////app/data/raw/mobility.db")
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# Extraction des outils spécifiques pour les nœuds
list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
run_query_tool = next(t for t in tools if t.name == "sql_db_query")

# Noeuds d'outils
get_schema_node = ToolNode([get_schema_tool])
run_query_node = ToolNode([run_query_tool])


# 1. Lister les tables (Ancrage initial)
def list_tables_node(state: MessagesState):
    content = f"Tables disponibles : {db.get_usable_table_names()}"
    return {"messages": [AIMessage(content=content)]}

# 2. Générer la requête avec le contexte RAG 
def generate_query(state: MessagesState):
    system_prompt = f"""Tu es un expert SQL pour Montréal. 
    Utilise ROUND(LOC_LAT, 3) pour les jointures entre collisions et requetes_311.
    Respecte le dictionnaire : 
    - collisions: sécurité routière.
    - requetes_311: signalements urbains.
    Limite à 5 résultats maximum. Ne fais que des SELECT."""
    
    llm_with_tools = model.bind_tools([run_query_tool])
    # On passe l'historique des messages pour le contexte
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}

# 3. Mode Contradicteur et Finalisation [cite: 83, 84]
def finalize_with_critic(state: MessagesState):
    last_message = state["messages"][-1].content
    prompt = f"""Analyse ces résultats SQL et réponds à la question initiale.
    TU DOIS INCLURE :
    - Une section 'Limites d'interprétation'[cite: 85].
    - Une section 'Ce que je vérifierais ensuite'[cite: 86].
    Données : {last_message}"""
    
    response = model.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal[END, "run_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "run_query"
    return END

builder = StateGraph(MessagesState)

builder.add_node("list_tables", list_tables_node)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query_node)
builder.add_node("finalizer", finalize_with_critic)

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "generate_query")

builder.add_conditional_edges(
    "generate_query",
    should_continue,
    {
        "run_query": "run_query",
        END: "finalizer"
    }
)

builder.add_edge("run_query", "generate_query")
builder.add_edge("finalizer", END)

graph = builder.compile()
if __name__ == "__main__":
    print("\n--- 🚀 TEST MOBILITY COPILOT (CLI MODE) ---")
    
    # On s'assure d'utiliser le bon nom d'objet (builder.compile() retourne l'app)
    app = builder.compile() 
    
    # Question de test basée sur les critères du hackathon (Météo + Collisions)
    question = "Quels types de requêtes 311 augmentent quand la température passe sous 0°C ?"
    
    inputs = {"messages": [HumanMessage(content=question)]}
    
    print(f" Question posée : {question}\n")

    # On itère sur les événements du graphe
    for event in app.stream(inputs, stream_mode="values"):
        message = event["messages"][-1]
        
        # 1. On affiche la réflexion SQL (Génération supervisée)
        if isinstance(message, AIMessage) and message.tool_calls:
            print(f" [RAISONNEMENT] Génération de la requête SQL...")
            print(f" SQL : {message.tool_calls[0]['args']['query']}")
            
        # 2. On affiche la réponse finale (Mode Contradicteur)
        elif isinstance(message, AIMessage) and not message.tool_calls:
            print(f"\n [RÉPONSE FINALE]")
            print("-" * 30)
            print(message.content)
            print("-" * 30)