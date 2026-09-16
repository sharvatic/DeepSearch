from typing import Annotated, List, TypedDict
import operator

class AgentState(TypedDict):
    user_query: str                # The user's original question
    search_queries: List[str] # Queries the LLM generates to search for
    search_results: Annotated[List[str], operator.add] # Raw text/snippets found online
    final_answer: str             # The final answer produced
    iteration_count: int          # To prevent infinite loops
    missing_report: str
    confidence_score: float
    loop_decision: str
