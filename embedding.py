import os
from  langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
class Embedding:
    def __init__(self,path_of_file):
        self.path_of_file = path_of_file
        pass
    def create_embedding(self):
        loader = TextLoader(self.path_of_file,encoding="utf-8")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            separator="",
            chunk_size=1200,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        index = os.getenv("PINECONE_INDEX_NAME")

        pinecone = PineconeVectorStore.from_documents(texts, embeddings,index_name=index)
        print("finished embedding")