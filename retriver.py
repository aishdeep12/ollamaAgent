import os
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import  init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings


load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)
llm = ChatOllama(model="qwen3:8b", temperature=0)

@tool(response_format="content_and_artifact")
def retrieve_relevant_context(query: str) -> str:
    """Search for relevant context to the query."""
    retriver = vector_store.as_retriever(search_kwargs={"k": 3})
    doc = retriver.invoke(query)
    serialized_docs =  "\n\n".join([
    f"Source: {doc.metadata.get('source', 'unknown')}\n\nContent: {doc.page_content}"
    for doc in doc])
    return serialized_docs,serialized_docs

def get_answer(query: str) -> str:
    """Use the retrieved context to answer the query."""
    agent = create_agent(model=llm, tools=[retrieve_relevant_context], system_prompt="You are an assistant that retrieves,always use the tool to answer else say i dont know")
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})
    return response


if __name__ == "__main__":
    query = "tell about tigris products and their features"
    answer = get_answer(query)
    print(f"Answer: {answer}")
