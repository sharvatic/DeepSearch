import uuid
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from Agent.main import AgentState
from Agent.main import HEADERS

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
