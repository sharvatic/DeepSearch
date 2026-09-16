import os
import sys
import asyncio
import httpx
# Ensure both Agent folder and root folder are in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END, START

try:
    from state import AgentState
except ImportError:
    from Agent.state import AgentState

try:
    from Nodes.printer import printer_node
    from Nodes.planner import planner
    from Nodes.researcher import researcher
    from Nodes.scraper import scraper_saver
    from Nodes.analyst import analyzer
    from Routers.loop_back import route_after_analyst
except ImportError:
    from Agent.Nodes.printer import printer_node
    from Agent.Nodes.planner import planner
    from Agent.Nodes.researcher import researcher
    from Agent.Nodes.scraper import scraper_saver
    from Agent.Nodes.analyst import analyzer
    from Agent.Routers.loop_back import route_after_analyst

# Build the Graph
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

app = workflow.compile()

async def main():
    initial_state = AgentState(
    user_query="What are different education policies for students in India and how do they compare to the US?", # Example question
    search_queries=[],
    search_urls=[],
    search_results=[],
    final_answer="",
    iteration_count=0
    )
    
    async for chunk in app.astream(initial_state, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            print(f"\n[Node Executed]: {node_name}")
            
            # target_variable = "iteration_count" 
            
            # if target_variable in state_update:
            #     print(f"Value of '{target_variable}' at this step:")
            #     print(state_update[target_variable])
            # else:
            #     print(f"'{target_variable}' did not change in this step.")


if __name__ == "__main__":
    asyncio.run(main())
    
print("\n--- Execution Finished ---")