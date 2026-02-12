from crewai import Task
from agents import paper_analyzer, result_summarizer, citation_agent, qa_agent

def create_tasks(paper_text):

    analyze_task = Task(
        description=f"""
        Analyze the following research paper:

        {paper_text}

        Extract:
        - Problem statement
        - Dataset used
        - Model architecture
        - Algorithm steps
        """,
        expected_output="Structured methodology explanation",
        agent=paper_analyzer
    )

    result_task = Task(
        description=f"""
        From the same paper:

        Extract:
        - Key results
        - Accuracy/performance
        - Contributions
        - Comparison with prior work
        """,
        expected_output="Structured result summary",
        agent=result_summarizer
    )

    citation_task = Task(
        description=f"""
        Extract references section and format in APA style.
        """,
        expected_output="APA formatted citations",
        agent=citation_agent
    )

    return analyze_task, result_task, citation_task
