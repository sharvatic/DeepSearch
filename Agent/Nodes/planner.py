try:
    from state import AgentState
    from models import llm
    from prompts import planner_prompt, planner_prompt_repeated
except ImportError:
    from Agent.state import AgentState
    from Agent.models import llm
    from Agent.prompts import planner_prompt, planner_prompt_repeated
from langchain_core.messages import SystemMessage, HumanMessage
import json


def planner(state: AgentState):
    print("---PLANNING SEARCH QUERIES---")

    task = state.get("user_query")
    
    it_count = state.get("iteration_count", 0)
    missing_report = state.get("missing_report", "No missing details reported.")
    planner_queries = state.get("search_queries", [])
    
    if it_count > 0:
        missing_report = state.get("missing_report", "No missing details reported.")
        formatted_prompt = planner_prompt_repeated.substitute(
            task=task,
            iteration_count=it_count,
            missing_report=missing_report,
            planner_queries=planner_queries
        )
    else:
        formatted_prompt = planner_prompt.substitute(task=task)
    
    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=task)
    ]
    
    llm_response = llm.invoke(messages)
    
    try:
        response_json = json.loads(llm_response.content)
        print(f"Planner LLM Response: {response_json}")

        return{
            "search_queries": [q["query"] for q in response_json.get("queries", [])]
        }
    except Exception as e:
        print(f"Error parsing Planner output: {e}")
        # Fallback logic: return a simplified version or retry
        return {"search_queries": [task]}