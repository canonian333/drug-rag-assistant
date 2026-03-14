import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tavily import TavilyClient
from langchain_core.documents import Document
from utils.vectorstore import similarity_search
from config.config import (
    TRAVILY_API,
    TRAVELY_MAX_RESULTS,
    RELEVANCE_THRESHOLD
)

def db_retrieve(query: str, index) -> tuple[list[Document], float]:
    try:
        if not query.strip():
            raise ValueError("Empty query provided")
        if index is None:
            raise ValueError("FAISS index is None")
        chunks, max_score = similarity_search(query, index)
        if not chunks:
            print("No results found in vector database")
            return [], 0.0
        print(f"DB retrieval successful")
        return chunks, max_score
    except ValueError as ve:
        print(f"DB retrieval error: {ve}")
        return [], 0.0
    except Exception as e:
        print(f"DB retrieval failed: {e}")
        return [], 0.0

def web_retrieve(query: str) -> list[dict]:
    try:
        if not query.strip():
            raise ValueError("Empty query provided")
        if not TRAVILY_API:
            raise ValueError("TRAVILY_API_KEY is missing")
        client  = TavilyClient(api_key=TRAVILY_API)
        results = client.search(
            query             = query,
            max_results       = TRAVELY_MAX_RESULTS,
            search_depth      = "advanced",       
            include_domains   = [               
                "webmd.com",
                "fda.gov",
                "mayoclinic.org",
                "drugs.com",
                "medlineplus.gov"
            ]
        )
        web_results = []
        for result in results.get("results", []):
            web_results.append({
                "title"   : result.get("title", "Unknown"),
                "content" : result.get("content", ""),
                "url"     : result.get("url", "")
            })
        if not web_results:
            print("No web results found")
            return []
        print(f"Web retrieval successful | {len(web_results)} results found")
        return web_results
    except ValueError as ve:
        print(f"Web retrieval error: {ve}")
        return []
    except Exception as e:
        print(f"Web retrieval failed: {e}")
        return []

def format_db_context(chunks: list[Document]) -> str:
    """Formats database chunks into a single context string."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        drug_name = chunk.metadata.get("drug_name", "Unknown")
        condition = chunk.metadata.get("condition", "Unknown")
        context_parts.append(
            f"[Source {i}] Drug: {drug_name} | Condition: {condition}\n"
            f"{chunk.page_content}"
        )
    return "\n\n".join(context_parts)

def format_web_context(web_results: list[dict]) -> str:
    """Formats web results into a single context string."""
    context_parts = []
    for i, result in enumerate(web_results, 1):
        context_parts.append(
            f"[Source {i}] {result['title']}\n"
            f"{result['content']}\n"
            f"URL: {result['url']}"
        )
    return "\n\n".join(context_parts)

def smart_retrieve(query: str, index) -> dict:
    try:
        chunks, best_distance = db_retrieve(query, index)
        # FAISS returns L2 distance.
        # Convert to similarity score s where 1.0 is a perfect match.
        similarity_score = 1 / (1 + best_distance) if chunks else 0.0
        if chunks and similarity_score >= RELEVANCE_THRESHOLD:
            print(f"DB threshold passed ({similarity_score:.4f} >= {RELEVANCE_THRESHOLD}) — using database")
            context  = format_db_context(chunks)
            metadata = [
                {
                    "drug_name" : chunk.metadata.get("drug_name", "Unknown"),
                    "condition" : chunk.metadata.get("condition", "Unknown"),
                    "rating"    : chunk.metadata.get("rating", "N/A"),
                    "score"     : round(similarity_score, 4)
                }
                for chunk in chunks
            ]
            return {
                "context"  : context,
                "source"   : "database",
                "metadata" : metadata
            }
        
        print(f"DB threshold not met ({similarity_score:.4f} < {RELEVANCE_THRESHOLD}) — falling back to web")
        web_results = web_retrieve(query)
        if not web_results:
            return {
                "context"  : "",
                "source"   : "none",
                "metadata" : []
            }
        context  = format_web_context(web_results)
        metadata = [
            {
                "title" : result["title"],
                "url"   : result["url"]
            }
            for result in web_results
        ]
        return {
            "context"  : context,
            "source"   : "web",
            "metadata" : metadata
        }

    except Exception as e:
        print(f"Smart retrieval failed: {e}")
        return {
            "context"  : "",
            "source"   : "none",
            "metadata" : []
        }

if __name__ == "__main__":
    from utils.vectorstore import index_load
    index = index_load()
    if index:
        test_queries = [
            ("DB possible hit", "What are the side effects of Guanfacine for ADHD?"),
            ("DB possible hit", "Is Valsartan effective for heart conditions?"),
            ("Web fallback",  "What are the side effects of Leqembi for Alzheimer's?"),
        ]

        for desc, query in test_queries:
            print(f"\n--- Testing: {desc} ---")
            print(f"Query    : {query}")
            result = smart_retrieve(query, index)
            print(f"Source   : {result['source']}")
            if result['metadata']:
                print(f"Metadata : {result['metadata'][0]}")
            print(f"Context  : {result['context'][:100]}")