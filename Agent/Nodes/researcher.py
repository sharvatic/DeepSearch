import requests
from Agent.main import AgentState


def researcher(state: AgentState):
    print("---RESEARCHING WEB---")

    SearXNG_URL = "http://localhost:8080/search"

    all_queries = state.get("search_queries", [])
    all_results = []

    for query in all_queries:
        try:
            response = requests.get(SearXNG_URL, params={"q": query, "format": "json", "lang": "us-en"}, timeout=10)
            if response.status_code == 200:
                print(f"Finding URLs for search query: '{query}'")
                data = response.json()
                results = data.get("results", [])
                if results:
                    urls = [result.get("url", "" ) for result in results]
                    all_results.extend(urls)

        except Exception as e:
            print(f"Error during web search for query '{query}': {e}")
            continue

    unique_urls = list(set(all_results))
    filtered_urls = [url for url in unique_urls if url]
    final_urls = filtered_urls[:7]
    print(f"Researcher found URLs: {final_urls}")

    return {"search_results": final_urls}