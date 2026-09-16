from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

target_collection = "research_docs"

client.delete_collection(collection_name=target_collection)

# List all existing collections
all_collections = [c.name for c in client.get_collections().collections]
print("All collections:", all_collections)