from crewai import Agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = "groq/llama-3.1-8b-instant"

paper_analyzer = Agent(
    role="Paper Analyzer",
    goal="Extract methodology, dataset, model architecture, and algorithm steps",
    backstory="Expert research scientist who deeply analyzes academic papers",
    llm=llm,
    verbose=True
)

result_summarizer = Agent(
    role="Result Summarizer",
    goal="Extract results, contributions, and experimental findings",
    backstory="Academic expert in evaluating research outcomes",
    llm=llm,
    verbose=True
)

citation_agent = Agent(
    role="Citation Agent",
    goal="Extract references and convert into APA format",
    backstory="Research assistant specialized in academic citations",
    llm=llm,
    verbose=True
)

qa_agent = Agent(
    role="Q&A Agent",
    goal="Answer doubts related to the research paper",
    backstory="Helpful professor answering student questions",
    llm=llm,
    verbose=True
)
