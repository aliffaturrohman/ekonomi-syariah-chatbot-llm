# graph.py
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List
from typing_extensions import TypedDict
import operator

llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)

class State(TypedDict):
    messages: Annotated[List, operator.add]

def call_model(state: State):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("chatbot", call_model)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

app_graph = workflow.compile()