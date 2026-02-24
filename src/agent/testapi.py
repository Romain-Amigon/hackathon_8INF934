import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


def test_groq_connection():
    try:
        llm = ChatGroq(
            model_name="moonshotai/kimi-k2-instruct-0905",
            temperature=0,
        )

        question = "Quels sont les avantages d'utiliser Groq pour un assistant de mobilité urbaine à Montréal ?"
        
        print(f"🚀 Envoi de la question à Groq...")
        response = llm.invoke([HumanMessage(content=question)])
        
        print("\n✅ Connexion réussie !")
        print(f"--- Réponse du modèle ---\n{response.content}")
        
    except Exception as e:
        print(f"\n❌ Erreur de connexion : {e}")

if __name__ == "__main__":
    print("testing conn")
    test_groq_connection()