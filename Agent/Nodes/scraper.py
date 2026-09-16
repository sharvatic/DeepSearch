import uuid
import asyncio
from typing import List, Dict, Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

try:
    from state import AgentState
    from config import COLLECTION_NAME
    from embeddings import get_embeddings
except ImportError:
    from Agent.state import AgentState
    from Agent.config import COLLECTION_NAME
    from Agent.embeddings import get_embeddings


# 1. Structure-Aware Parent Splitter: Splits along Markdown headers
headers_to_split_on = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
]
markdown_parent_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=False
)

parent_safety_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". "]
)

# 2. Granular Child Splitter: Sub-chunks parent sections into precise vector targets
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)


async def scrape_and_save_crawl4ai(state: AgentState) -> AgentState:
    print("--- SCRAPING AND SAVING (CRAWL4AI + QDRANT) ---")
    
    urls: List[str] = state.get("search_results", [])
    if not urls:
        print("No URLs found in state to scrape.")
        return state

    # 1. Initialize Qdrant and Embeddings
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    embeddings_model = get_embeddings()

    sample_vector = embeddings_model.embed_documents(["init_check"])[0]
    vector_dim = len(sample_vector)

    try:
        client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    # 2. Configure Crawl4AI
    browser_cfg = BrowserConfig(
        headless=True,
        verbose=False
    )
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=20,     # Discard empty or error pages
        remove_overlay_elements=True # Automatically drop popups / cookie banners
    )

    scraped_documents = []

    # 3. Concurrent Web Crawl via AsyncWebCrawler
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        print(f"Crawling {len(urls)} URLs in parallel...")
        crawl_results = await crawler.arun_many(urls=urls, config=run_cfg)

        for res in crawl_results:
            if res.success and res.markdown:
                scraped_documents.append({
                    "url": res.url,
                    "markdown": res.markdown
                })
            else:
                print(f"Failed or blocked crawl for: {res.url} | Error: {res.error_message}")

    if not scraped_documents:
        print("All crawls failed or returned empty content.")
        return state

    # 4. Hierarchical Parent-Child Chunking
    prepared_chunks: List[Dict[str, Any]] = []

    for doc in scraped_documents:
        url = doc["url"]
        markdown_text = doc["markdown"]

        # Split into section-level parents
        parent_docs = markdown_parent_splitter.split_text(markdown_text)

        # Fallback if page lacks markdown headers
        if not parent_docs:
            fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            parent_chunks = fallback_splitter.split_text(markdown_text)
        else:
            parent_chunks = [p.page_content for p in parent_docs]

        for sec_idx, parent_text in enumerate(parent_chunks):
            # Deterministic Parent ID
            parent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}_p_{sec_idx}"))
            
            # Split parent into small child chunks for vector retrieval
            child_chunks = child_splitter.split_text(parent_text)

            for child_idx, child_text in enumerate(child_chunks):
                # Deterministic Child ID based on content to prevent duplicate vectors
                child_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{url}_{parent_id}_{child_text}"))
                
                prepared_chunks.append({
                    "child_id": child_id,
                    "child_text": child_text,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "url": url,
                    "child_index": child_idx
                })

    if not prepared_chunks:
        print("No valid chunks extracted.")
        return state

    # 5. Batch Vector Generation (Massive Speedup over single calls)
    # 5. Batch Vector Generation
    batch_size = 64
    vectors = []
    child_texts = [item["child_text"] for item in prepared_chunks]
    total_chunks = len(child_texts)

    print(f"Embedding {total_chunks} child chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch = child_texts[i : i + batch_size]
        try:
            # Safely process 64 chunks per network call
            batch_vectors = embeddings_model.embed_documents(batch)
            vectors.extend(batch_vectors)
            print(f"Embedded {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")
        except Exception as e:
            print(f"[Embedding Error] Batch failed at index {i}: {e}")
            raise e

    # 6. Build Qdrant PointStructs
    points = [
        PointStruct(
            id=item["child_id"],
            vector=vector,
            payload={
                "parent_id": item["parent_id"],
                "parent_text": item["parent_text"],
                "child_text": item["child_text"],
                "url": item["url"],
                "child_index": item["child_index"]
            }
        )
        for item, vector in zip(prepared_chunks, vectors)
    ]

    # 7. Upsert to Qdrant in Batches (Default batch size: 100)
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"Successfully saved {len(points)} child chunks to Qdrant.")
    return state


# Synchronous wrapper if your LangGraph node runner requires a sync function
def scraper_saver(state: AgentState):
    return asyncio.run(scrape_and_save_crawl4ai(state))