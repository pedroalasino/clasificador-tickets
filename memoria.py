import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=os.environ["GEMINI_API_KEY"],
)

def responder(estado):
    return {"messages": [llm.invoke(estado["messages"])]}

grafo = StateGraph(MessagesState)
grafo.add_node("responder", responder)
grafo.add_edge(START, "responder")
app = grafo.compile(checkpointer=InMemorySaver())

def hablar(thread_id, mensaje):
    config = {"configurable": {"thread_id": thread_id}}
    resultado = app.invoke({"messages": [("user", mensaje)]}, config=config)
    print(f"[{thread_id}] {resultado['messages'][-1].text}\n")

hablar("cliente-1", "Hola, me llamo Pedro y mi pedido es el 123.")
hablar("cliente-1", "¿Cómo me llamo y cuál es mi pedido?")
hablar("cliente-2", "¿Cómo me llamo y cuál es mi pedido?")