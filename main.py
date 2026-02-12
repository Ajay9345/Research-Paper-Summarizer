from crewai import Crew
from pdf_loader import extract_text_from_pdf
from tasks import create_tasks
from agents import qa_agent
from crewai import Task

def run_summary(pdf_path):
    paper_text = extract_text_from_pdf(pdf_path)

    analyze_task, result_task, citation_task = create_tasks(paper_text)

    crew = Crew(
        agents=[analyze_task.agent,
                result_task.agent,
                citation_task.agent],
        tasks=[analyze_task,
               result_task,
               citation_task],
        verbose=True
    )

    return crew.kickoff()


def run_qa(question, paper_text):
    qa_task = Task(
        description=f"""
        Paper Content:
        {paper_text}

        Student Question:
        {question}
        """,
        expected_output="Clear explanation",
        agent=qa_agent
    )

    crew = Crew(
        agents=[qa_agent],
        tasks=[qa_task]
    )

    return crew.kickoff()


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")

    print("\nGenerating Summary...\n")
    summary = run_summary(pdf_path)
    print(summary)

    while True:
        question = input("\nAsk a doubt (type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = run_qa(question, extract_text_from_pdf(pdf_path))
        print(answer)
