from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen2.5:1.5b",
    num_ctx=8192,
    num_predict=768,
    temperature=0,
    format="json",
)
