import os
import asyncio
from typing import List, Dict, Any
from tavily import AsyncTavilyClient
try:
    from state import AgentState
except ImportError:
    from Agent.state import AgentState


tavily_client = AsyncTavilyClient(api_key=os.environ.get("TAVILY_SEARCH_KEY"))

async def search_single_query(query: str, max_results: int = 5) -> List[str]:
    """
    Executes a single search query on Tavily and returns the top URLs.
    """
    try:
        response = await tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False
        )
        results = response.get("results", [])
        return [r["url"] for r in results if r.get("url")]
    except Exception as e:
        print(f"[Tavily Warning] Failed search for query '{query}': {e}")
        return []


async def researcher(state: AgentState) -> Dict[str, List[str]]:
    print("---RESEARCHING WEB (TAVILY)---")
    
    search_queries = state.get("search_queries", [])
    print(f"Researcher received queries: {search_queries}")
    
    if not search_queries:
        return {"search_results": []}

    # Fetch top 5 URLs for all search queries concurrently
    tasks = [search_single_query(query, max_results=5) for query in search_queries]
    query_results: List[List[str]] = await asyncio.gather(*tasks)

    # Flatten and deduplicate within this step while preserving relevance order
    seen = set()
    new_urls: List[str] = []
    
    for url_list in query_results:
        for url in url_list:
            if url not in seen:
                seen.add(url)
                new_urls.append(url)

    print(f"Discovered {len(new_urls)} new URLs across all subqueries.")

    # Because search_results uses operator.add in AgentState, 
    # returning this list will automatically append them to the existing state.
    return {"search_results": new_urls}