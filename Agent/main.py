from typing import Annotated, List, TypedDict
import operator
import json, requests
import uuid
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompts import planner_prompt, analyst_prompt, planner_prompt_repeated


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


# 2. Define the Logic (The Nodes)
def planner(state: AgentState):
    print("---PLANNING SEARCH QUERIES---")

    task = state.get("user_query")
    
    it_count = state.get("iteration_count", 0)
    missing_report = state.get("missing_report", "No missing details reported.")
    planner_queries = state.get("planner_queries", [])
    
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




def scraper_saver(state: AgentState):
    print("---SCRAPING AND SAVING---")
    
    urls = state.get("search_results", [])
    if not urls:
        print("No URLs to scrape.")
        return state

    client = QdrantClient(url="http://localhost:6333", timeout=60)
    collection_name = "research_docs"

    try:
        client.get_collection(collection_name=collection_name)
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE), # 1536  is small because we don't need large dim as our text is not gon abe that ;large as we are using the parent child model of embeddings
        )

    embeddings_model = OllamaEmbeddings(model="qwen2.5:1.5b") 

    points = []

    # Define Parent Splitter (The full context)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    # Define Child Splitter (The granular search target)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    
    for url in urls:
        try:
            print(f"Scraping: {url}")
            res = requests.get(url, headers=HEADERS, timeout=5)
            res.raise_for_status()
            
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # Clean up: remove script and style elements
            for s in soup(["script", "style", "nav", "footer"]): s.decompose()
            full_text = " ".join(soup.get_text(separator=' ').split())

            # Step 1: Break webpage into large PARENT chunks
            parent_chunks = parent_splitter.split_text(full_text)

            for parent_text in parent_chunks:
                parent_id = str(uuid.uuid4())  # Unique ID for the parent chunk

                # Step 2: Break this specific parent down into tiny CHILD chunks
                child_chunks = child_splitter.split_text(parent_text)

                for child_text in child_chunks:
                    vector = embeddings_model.embed_query(child_text)
                    child_id = str(uuid.uuid4())  # Unique ID for the child chunk

                    points.append(PointStruct(
                        id=child_id,
                        vector=vector,
                        payload={
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                            "child_text": child_text,
                            "url": url
                        }
                    ))

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            continue

    if points:
        try:
            client.upsert(collection_name=collection_name, points=points)
            print(f"Successfully saved {len(points)} documents to Qdrant.")
        except Exception as e:
            print(f"Failed to save documents to Qdrant by Sharv: {e}")

    # Return updated state (adding raw text for the Analyst to see immediately)
    return state



def analyzer(state: AgentState):
    print("---ANALYZING RESULTS---")
    # This is where your analyze_results function will live
    
    print("--- EXECUTING ANALYST & ITERATION NODE ---")
    
    user_query = state.get("user_query", "")
    planner_queries = state.get("planner_queries", []) # Queries generated by planner node
    iteration_count = state.get("iteration_count", 0)
    
    # 1. Gather Context from Qdrant using the Planner's Queries
    client = QdrantClient(url="http://localhost:6333")
    embeddings_model = OllamaEmbeddings(model="qwen2.5:1.5b")
    collection_name = "research_docs"
    
    retrieved_chunks = []
    
    for query in planner_queries:
        query_vector = embeddings_model.embed_query(query)
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=2
        )
        
        for result in search_results:
            payload = result.payload
            retrieved_chunks.append(payload.get('parent_text', ''))
            
    unified_context = "\n\n---\n\n".join(list(set(retrieved_chunks)))
    
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0, format="json")
    
    
    formatted_analyst_prompt = analyst_prompt.substitute(
        user_query=user_query,
        unified_context=unified_context,
        iteration_count=iteration_count
    )
    try:
        response = llm.invoke(formatted_analyst_prompt)
        analysis_json = json.loads(response.content)
        
        print(f"Analyst confidence: {analysis_json['confidence_score']}%")
        print(f"-> Next Action: {analysis_json['loop_decision']}")
        
        return {
                "final_answer": analysis_json["current_summary"],
                "missing_report": analysis_json["missing_details_report"],
                "confidence_score": int(analysis_json["confidence_score"]),
                "loop_decision": analysis_json["loop_decision"],
                "iteration_count": iteration_count + 1 # Increment execution flag counter
            }
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {
            "final_answer": "Analysis failed. No summary available.",
            "missing_report": "Unable to determine missing details due to analysis failure.",
            "confidence_score": 0,
            "loop_decision": "TERMINATE",
        }


def route_after_analyst(state):
    print("--- ROUTING NEXT STEP ---")
    decision = state.get("loop_decision")
    iteration_count = state.get("iteration_count", 0)
    
    if decision == "CONTINUE" or iteration_count < 1:
        print("-> Decision: CONTINUE. Looping back to the Planner.")
        return "goToPlanner"
    else:
        print("-> Decision: TERMINATE. Sending data to the Printer Node.")
        return "goToPrinter"



def printer_node(state):
    print("\n==================================================")
    print("               FINAL RESEARCH REPORT              ")
    print("==================================================\n")
    
    print(state.get("final_answer", "No answer generated."))
    
    print("\n==================================================")
    print(f"Final Confidence Score: {state.get('confidence_score')}%")
    print(f"Total Iterations: {state.get('iteration_count')}")
    print("==================================================")
    
    return state



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
workflow.add_edge("analyst", "printer")

# workflow.add_conditional_edges(
#     "analyst",         # The node the graph stops at to check the condition
#     route_after_analyst,    # The routing function that reads the state
#     {
#         "goToPlanner": "planner", # Maps the router's string output to the actual node name
#         "goToPrinter": "printer"
#     }
# )

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