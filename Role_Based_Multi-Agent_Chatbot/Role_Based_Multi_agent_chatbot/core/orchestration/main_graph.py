import os
import datetime
from typing import TypedDict, List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langsmith import traceable # pylint: disable=import-error
from langgraph.graph import StateGraph, END # pylint: disable=import-error

# Agent imports
from agentic_student_assistant.talk2jobs.agents.job_market_agent import run_job_market_agent
# from agentic_student_assistant.core.base.books_agent import run_books_agent
from agentic_student_assistant.talk2books.agents.books_recommend_agent import BooksRecommendAgent
from agentic_student_assistant.talk2papers.agents.paper_recommend_agent import PaperRecommendAgent
from agentic_student_assistant.talk2docs.agents.docs_agent import DocsRecommendAgent
from agentic_student_assistant.core.base.fallback_agent import FallbackAgent
from agentic_student_assistant.core.orchestration.router_agent import route_query

load_dotenv()

# State definition
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: User query
        chat_history: Conversation history
        agent: The agent that handled the query
        result: The final answer
        confidence: Router confidence score
        reasoning: Router reasoning
        metadata: Additional metadata
    """
    query: str
    chat_history: List[str]
    agent: str
    result: str
    confidence: Optional[float]
    reasoning: str
    metadata: Optional[Dict[str, Any]]

# ------------ ROUTING (LLM-BASED) -------------

def route_agent(state: GraphState):
    """
    Route query using LLM-based semantic routing.
    Replaces keyword-based routing with GPT-powered understanding.
    """
    query = state["query"]
    
    # Use LLM router for intelligent routing with orchestration
    chat_history = state.get("chat_history", [])
    
    # Format history string if list provided (simple heuristic)
    history_str = str(chat_history)[-1000:] if chat_history else ""
    
    # Use LLM router
    decision = route_query(query, enable_orchestration=True, chat_history=history_str)
    
    print(f"🧭 Routing to: {decision.agent} agent")
    print(f"🎯 Confidence: {decision.confidence:.2f}")
    print(f"💭 Reasoning: {decision.reasoning}")
    
    return {
        "agent": decision.agent,
        "query": state["query"],
        "chat_history": state.get("chat_history", []),
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "metadata": {
            "router_version": "llm_v1",
            "model": "gpt-4"
        }
    }

# ------------ NODE DEFINITIONS -------------

@traceable(name="job_market_node")
def job_market_node(state: GraphState):
    result = run_job_market_agent(state["query"])  # ✅ Use single entry point
    return {"result": result, "agent": "job_market"}

@traceable(name="books_node")
def books_node(state: GraphState):
    agent = BooksRecommendAgent()
    result = agent.process(state["query"])
    return {"result": result, "agent": "books"}

@traceable(name="papers_node")
def papers_node(state: GraphState):
    agent = PaperRecommendAgent()
    result = agent.process(state["query"], chat_history=state.get("chat_history", []))
    return {"result": result, "agent": "papers"}

@traceable(name="fallback_node")
def fallback_node(state: GraphState):
    fallback = FallbackAgent()
    result = fallback.run(state["query"])
    return {"result": result, "agent": "fallback"}

@traceable(name="documents_node")
def documents_node(state: GraphState):
    """Node for document Q&A agent."""
    agent = DocsRecommendAgent()
    result = agent.process(state["query"], chat_history=state.get("chat_history", []))
    return {"result": result, "agent": "documents"}

@traceable(name="orchestrator_node")
def orchestrator_node(state: GraphState):
    """Node for orchestrator agent."""
    from agentic_student_assistant.core.orchestration.orchestrator_agent import OrchestratorAgent # pylint: disable=import-outside-toplevel
    
    orchestrator = OrchestratorAgent()
    result = orchestrator.process(state["query"])
    return {"result": result, "agent": "orchestrator"}

# ------------ GRAPH SETUP -------------
graph = StateGraph(GraphState)

graph.add_node("router", RunnableLambda(route_agent))
graph.add_node("job_market", RunnableLambda(job_market_node))
graph.add_node("books", RunnableLambda(books_node))
graph.add_node("papers", RunnableLambda(papers_node))
graph.add_node("documents", RunnableLambda(documents_node))
graph.add_node("fallback", RunnableLambda(fallback_node))
graph.add_node("orchestrator", RunnableLambda(orchestrator_node))  # NEW

graph.add_conditional_edges(
    "router",
    lambda state: state["agent"],
    {
        "job_market": "job_market",
        "books": "books",
        "papers": "papers",
        "documents": "documents",
        "orchestrator": "orchestrator",  # NEW
        "fallback": "fallback",
    },
)

graph.add_edge("job_market", END)
graph.add_edge("books", END)
graph.add_edge("papers", END)
graph.add_edge("documents", END)
graph.add_edge("orchestrator", END)  # NEW
graph.add_edge("fallback", END)

graph.set_entry_point("router")
app = graph.compile()

def log_query(query: str, agent: str, result: str, latency: float = None, is_fallback: bool = False, 
              confidence: float = None, reasoning: str = ""): # pylint: disable=R0917
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/workflow_logs.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"🕒 Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"❓ Query: {query}\n")
        f.write(f"📌 Routed Agent: {agent}\n")
        if confidence is not None:
            f.write(f"🎯 Router Confidence: {confidence:.2f}\n")
        if reasoning:
            f.write(f"💭 Router Reasoning: {reasoning}\n")
        if latency is not None:
            f.write(f"⏱️ Latency: {latency:.2f} seconds\n")
        f.write(f"🛡️ Fallback Used: {'Yes' if is_fallback else 'No'}\n")
        f.write("📘 Final Answer:\n")
        f.write(result + "\n")
        f.write("=" * 60 + "\n")


# ------------ CLI EXECUTION -------------
if __name__ == "__main__":
    user_query = input("\n🔎 Enter your query: ")
    final_state = app.invoke({
        "query": user_query,
        "chat_history": []
    })

    print(f"\n✅ Final Answer from {final_state['agent']} Agent:\n{final_state['result']}")
    
    log_query(
        query=user_query,
        agent=final_state["agent"],
        result=final_state["result"],
        confidence=final_state.get("confidence"),
        reasoning=final_state.get("reasoning", "")
    )
