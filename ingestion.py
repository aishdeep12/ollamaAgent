import asyncio
import os
from pyclbr import Class
from dotenv import load_dotenv
import ssl
from typing import Any,Dict,List
import logging

import certifi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl,TavilyExtract,TavilyMap

logger = logging.getLogger(__name__)

class Ingestion():

    def __init__(self):

        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.pinecone = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=self.embeddings)
        self.tavily_extract = TavilyExtract()
        self.tavily_crawl = TavilyCrawl()
        self.tavily_map = TavilyMap(max_depth=2, max_pages=100,max_breadth = 200)

    def ingest_url(self,url):
        res = self.tavily_crawl.invoke({
            "url": url,
            "max_depth": 1,
            "extract_depth":"advanced",
            "instructions":"Extract only product related information"
        })
        return res

    def chunking(self,content):
        log_info = f"Starting chunking process for content of length {len(content)}"
            # Reduce chunk size and overlap to avoid context length errors
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(content)
        log_info += f"Finished chunking process. Generated {len(chunks)} chunks."
        return chunks
    
    async def index_documents(self,documents,batch_size=5):
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        logger.info(f"number of batches to process: {len(batches)}")

        async def process_batch(batch: List[Document], batch_number: int):
            try:
                self.pinecone.add_documents(batch)
                logger.info(f"Successfully indexed batch of size {len(batch)}")
            except Exception as e:
                logger.error(f"Error indexing batch: {e} of batch number: {batch_number}")

        tasks = [process_batch(batch, idx+1) for idx, batch in enumerate(batches)]
        result = await asyncio.gather(*tasks, return_exceptions=True)

    

if __name__ == "__main__":
    ingestor = Ingestion()
    all_docs = ingestor.ingest_url(url="https://tigrislifesciences.in/")
    all_docs  = [Document(page_content=result['raw_content'], metadata = {"source": result['url']}) for result in all_docs['results']]
    asyncio.run(ingestor.index_documents(all_docs))
    print(all_docs)

    
