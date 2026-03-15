import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from models.llm import get_llm
from utils.retriever import smart_retrieve
from config.config import (
    CONCISE_INSTRUCTION,
    DETAILED_INSTRUCTION,
)

#Prompt templates
CONCISE_TEMPLATE = """
You are a medical information assistant specializing in drug information.
Your task is to answer the patient's question using ONLY the context provided below.

Instructions: {instruction}

Important Rules:
- Never hallucinate drug names, dosages, or side effects
- If the context does not contain enough information, say "I don't have enough information to answer this accurately"
- Never recommend or prescribe drugs
- Always remind the user to consult a healthcare professional

Context:
{context}

Question: {question}

Answer:
"""

DETAILED_TEMPLATE = """
You are a medical information assistant specializing in drug information.
Your task is to answer the patient's question using ONLY the context provided below.

Instructions: {instruction}

Important Rules:
- Never hallucinate drug names, dosages, or side effects
- If the context does not contain enough information, say "I don't have enough information to answer this accurately"
- Never recommend or prescribe drugs
- Structure your answer with clear sections where applicable
- Always remind the user to consult a healthcare professional

Context:
{context}

Question: {question}

Answer:
"""

def get_prompt(mode: str) -> PromptTemplate:
    try:
        if mode == "Concise":
            template    = CONCISE_TEMPLATE
            instruction = CONCISE_INSTRUCTION
        else:
            template    = DETAILED_TEMPLATE
            instruction = DETAILED_INSTRUCTION

        prompt = PromptTemplate(
            input_variables = ["context", "question"],
            partial_variables = {"instruction": instruction},
            template        = template
        )
        return prompt
    except Exception as e:
        print(f"Failed to build prompt: {e}")
        return None

#response generation
def generate_answer(query: str, context: str, mode: str) -> str:
    try:
        if not query.strip():
            raise ValueError("Empty query provided")
        if not context.strip():
            raise ValueError("Empty context provided")
        #prompt and llm
        llm    = get_llm(mode=mode)
        prompt = get_prompt(mode=mode)
        if llm is None:
            raise ValueError("LLM failed to initialize")
        if prompt is None:
            raise ValueError("Prompt failed to initialize")
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(f"\n Generating {mode} answer")

        answer = chain.invoke({
            "context" : context,
            "question": query
        })
        if not answer.strip():
            raise ValueError("LLM returned empty answer")
        print(f"Answer generated successfully")
        return answer.strip()
    except ValueError as ve:
        print(f"Generation error: {ve}")
        return None
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return None

#full rag pipeline
def run_rag_pipeline(query: str, index, mode: str = "Concise") -> dict:
    try:
        print(f"\n{'='*50}")
        print(f" Query : {query}")
        print(f"Mode  : {mode}")
        print(f"{'='*50}")

        retrieval = smart_retrieve(query, index)
        source   = retrieval["source"]
        context  = retrieval["context"]
        metadata = retrieval["metadata"]
        #handle no results
        if source == "none" or not context.strip():
            return {
                "answer"   : "I was unable to find relevant information for your query. Please consult a healthcare professional.",
                "source"   : "none",
                "metadata" : [],
                "mode"     : mode
            }

        #generate answer
        answer = generate_answer(
            query   = query,
            context = context,
            mode    = mode
        )
        if not answer:
            return {
                "answer"   : "I was unable to generate an answer. Please try again.",
                "source"   : source,
                "metadata" : metadata,
                "mode"     : mode
            }
        return {
            "answer"   : answer,
            "source"   : source,
            "metadata" : metadata,
            "mode"     : mode
        }
    except Exception as e:
        print(f"RAG pipeline failed: {e}")
        return {
            "answer"   : "An error occurred. Please try again.",
            "source"   : "none",
            "metadata" : [],
            "mode"     : mode
        }
        
if __name__ == "__main__":
    from utils.vectorstore import get_or_build_index
    index = get_or_build_index()
    if index:
        test_cases = [
            # DB hit — Detailed
            ("What are the side effects of Guanfacine for ADHD?","Detailed" ),
            # DB hit — Concise
            ("What are the side effects of NuvaRing for Birth control?", "Concise"),
            # Web fallback — Detailed
            ("What are the side effects of Leqembi for Alzheimer's?", "Concise" ),
        ]

        for query, mode in test_cases:
            print(f"\n{'='*50}")
            result = run_rag_pipeline(query, index, mode)
            print(f"Answer:\n{result['answer']}")
            print(f"Source: {result['source']}")
            print(f"Mode: {result['mode']}")
            print(f"Metadata: {result['metadata']}")
            print(f"{'='*50}")