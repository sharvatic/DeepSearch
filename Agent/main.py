from typing import Annotated, List, TypedDict
import operator
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from Nodes.printer import printer_node
from Nodes.planner import planner
from Nodes.researcher import researcher
from Nodes.scraper import scraper_saver
from Nodes.analyst import analyzer
from Router.loop_back import route_after_analyst



HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# 1. Define the State
# This dictionary keeps track of everything the agent knows during the search
class AgentState(TypedDict):
    user_query: str                # The user's original question
    search_queries: List[str] # Queries the LLM generates to search for
    search_results: Annotated[List[str], operator.add] # Raw text/snippets found online
    final_answer: str             # The final answer produced
    iteration_count: int          # To prevent infinite loops
    missing_report: str
    confidence_score: int
    loop_decision: str


llm = ChatOllama(model="qwen2.5:1.5b", num_ctx=2048, temperature=0, format="json")




# 3. Build the Graph
workflow = StateGraph(AgentState)

# Add our nodes
workflow.add_node("planner", planner)
workflow.add_node("search_web", researcher)
workflow.add_node("scraper_saver", scraper_saver)
workflow.add_node("analyst", analyzer)
workflow.add_node("printer", printer_node)

# Set the edges (The "Paths")
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "search_web")
workflow.add_edge("search_web", "scraper_saver")
workflow.add_edge("scraper_saver", "analyst")
# workflow.add_edge("analyst", "printer")

workflow.add_conditional_edges(
    "analyst",         # The node the graph stops at to check the condition
    route_after_analyst,    # The routing function that reads the state
    {
        "goToPlanner": "planner", # Maps the router's string output to the actual node name
        "goToPrinter": "printer"
    }
)

workflow.add_edge("printer", END)

# Compile the graph

initial_state = AgentState(
    user_query="What are different education policies for students in India and how do they compare to the US?", # Example question
    search_queries=[],
    search_results=[],
    final_answer="",
    iteration_count=0
)
app = workflow.compile()

for chunk in app.stream(initial_state, stream_mode="updates"):
    for node_name, state_update in chunk.items():
        print(f"\n[Node Executed]: {node_name}")
        
        # Replace 'your_variable_name' with the actual state key you want to track
        target_variable = "iteration_count" 
        
        if target_variable in state_update:
            print(f"Value of '{target_variable}' at this step:")
            print(state_update[target_variable])
        else:
            print(f"'{target_variable}' did not change in this step.")

print("\n--- Execution Finished ---")