from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from promptbuilder import prompt
from schemas import SummaryInput, SummaryOutput
from huggingface import HuggingFaceClient

load_dotenv()

def main():
    print("Hello from langchain!")
    print("Your API key is:", os.getenv("OPENAI_API_KEY"))
    information = """ Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of February 2026, Forbes estimates his net worth to be around US$852 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals."""

    summary_template = """Summarize the following information about Elon Musk: {information}
        1. A brief summary of who Elon Musk is.
        2. 2 interesting facts about Elon Musk.
        write in not more than five lines in total
        """
    inputs = input("What do you want to make content on? ")
    # summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    summary_prompt_template = PromptTemplate(
        input_variables=["topic", "number_of_paragraphs", "number_of_images", "tone", "additional_context"],
        template=prompt
    )
    llm = ChatOllama(model="qwen3:8b", temperature=0)
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

    @tool
    def  search_web(query: str) -> str:
        """Search the web for queries related to current facts, biographies, ages, Wikipedia-style lookups, or web facts."""
        search = TavilySearch()
        return search.run(query)
    
    toolkit = [search_web]
    agent = create_agent(model=llm, tools=toolkit, system_prompt=( f"Do a websearch when triggered by user"))
    agent_response = agent.invoke({"messages": [HumanMessage(content=f"Find on web SEO keywords for: {inputs}")]})
    print(f"Search Agent Response: {agent_response}")

    print("Now generating images based on prompts...")
    hf_client = HuggingFaceClient()
    for image_key, image_prompt in response.image_prompts.items():
        hf_client.generate_image(image_prompt, image_key)


    # inputs = input("What do you want to ask? ")



if __name__ == "__main__":
    main()
