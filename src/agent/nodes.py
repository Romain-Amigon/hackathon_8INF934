import asyncio
import nest_asyncio
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent

nest_asyncio.apply()

df_311 = pd.read_csv("../../data/raw/requetes_311.csv", low_memory=False)
df_coll = pd.read_csv("../../data/raw/collisions_clean.csv")
df_meteo = pd.read_csv("../../data/raw/weather_montreal.csv")

llm = Ollama(model="llama3", request_timeout=3000.0)

instruction_stricte = """\
You are working with a pandas DataFrame named `df`.
1. Convert the query to executable Python code using pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code must ALWAYS start with the variable `df`.
4. PRINT ONLY THE EXPRESSION.
5. DO NOT RETURN ANY TEXT, EXPLANATIONS, OR MARKDOWN. ONLY THE RAW PYTHON EXPRESSION.
"""

engine_311 = PandasQueryEngine(df=df_311, llm=llm, instruction_str=instruction_stricte)
engine_coll = PandasQueryEngine(df=df_coll, llm=llm, instruction_str=instruction_stricte)
engine_meteo = PandasQueryEngine(df=df_meteo, llm=llm, instruction_str=instruction_stricte)

outil_311 = QueryEngineTool(
    query_engine=engine_311,
    metadata=ToolMetadata(
        name="donnees_311",
        description="Utilise cet outil exclusivement pour répondre aux questions concernant les requêtes citoyennes 311 (nids-de-poule, déneigement, feux de circulation, etc.)."
    )
)

outil_coll = QueryEngineTool(
    query_engine=engine_coll,
    metadata=ToolMetadata(
        name="donnees_collisions",
        description="Utilise cet outil exclusivement pour obtenir des statistiques sur les accidents de la route, les blessés et les collisions."
    )
)

outil_meteo = QueryEngineTool(
    query_engine=engine_meteo,
    metadata=ToolMetadata(
        name="donnees_meteo",
        description="Utilise cet outil exclusivement pour obtenir des informations sur la température, la neige et la pluie."
    )
)

agent_config = ReActAgent(
    name="agent_donnees",
    description="Agent pour répondre aux questions sur la mobilité.",
    tools=[outil_311, outil_coll, outil_meteo],
    llm=llm
)

workflow = AgentWorkflow(agents=[agent_config], root_agent="agent_donnees")

async def main():
    reponse = await workflow.run(user_msg="Combien de nids-de-poule sont signalés dans le dataset 311, et combien y a-t-il de morts dans le dataset des collisions ?")
    print(str(reponse))

if __name__ == "__main__":
    asyncio.run(main())