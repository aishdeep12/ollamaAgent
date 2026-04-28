from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from embedding import Embedding
from promptbuilder import prompt,rag_prompt
from schemas import SummaryInput, SummaryOutput
from huggingface import HuggingFaceClient

load_dotenv()

def main():
    print("Hello from langchain!")
    print("Your API key is:", os.getenv("OPENAI_API_KEY"))
    inputs = input("What do you want to make content on? ")
    llm = ChatOllama(model="qwen3:8b", temperature=0)

    def content_creation(llm):
    
        summary_prompt_template = PromptTemplate(
            input_variables=["topic", "number_of_paragraphs", "number_of_images", "tone", "additional_context"],
            template=prompt
        )
        
        structured_llm = llm.with_structured_output(SummaryOutput)
        chain = summary_prompt_template | structured_llm
        user_input = SummaryInput(
        topic=inputs,
        number_of_paragraphs=2,
        number_of_images=2,
        tone="professional"
    )
        response = chain.invoke(user_input.model_dump())
        print(response)
        return response

    @tool
    def  search_web(query: str) -> str:
        """Search the web for queries related to current facts, biographies, ages, Wikipedia-style lookups, or web facts."""
        search = TavilySearch()
        return search.run(query)
    
    def agentic_tool_search(query: str,llm,response) -> str:
        """Search the web for queries related to current facts, biographies, ages, Wikipedia-style lookups, or web facts."""
        toolkit = [search_web]
        agent = create_agent(model=llm, tools=toolkit, system_prompt=( f"Do a websearch when triggered by user"))
        agent_response = agent.invoke({"messages": [HumanMessage(content=f"Find on web SEO keywords for: {query}")]})
        print(f"Search Agent Response: {agent_response}")

        print("Now generating images based on prompts...")
        hf_client = HuggingFaceClient()
        for image_key, image_prompt in response.image_prompts.items():
            hf_client.generate_image(image_prompt, image_key)
        ## temporary return needs to be adjusted to prduce final good response 
        return agent_response
    

    ###############################################################
    #Context oriented search and prompt building using RAG
    ###############################################################

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def context_search(query: str) -> str:
        """Search for relevant context to the query."""
        # This is where you would implement your context search logic, e.g., using a vector database or a search engine.
        vector_store = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=OllamaEmbeddings(model="nomic-embed-text"))
        retriver = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriver.invoke(query)
        return format_docs(docs)
    
    # embedding = Embedding(path_of_file=os.path.join(os.path.dirname(__file__), "book.txt"))
    # embedding.create_embedding()
    rag_prompt_template = ChatPromptTemplate.from_template(rag_prompt)
    llm_query = rag_prompt_template.format_messages(context=context_search(inputs), question=inputs)
    llm_response = llm.invoke(llm_query)
    print(f"RAG Response: {llm_response}")



if __name__ == "__main__":
    main()
