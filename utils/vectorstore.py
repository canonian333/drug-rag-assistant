import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from models.embeddings import create_embeddings
from config.config import FAISS_INDEX_PATH

def create_index(chunks: list[Document])-> FAISS:
    try:
        if not chunks:
            raise ValueError("No chunks created")
        print(f"creating index with {len(chunks)} chunks")
        embeddings=create_embeddings()
        if embeddings is None:
            raise ValueError("Embeddings are not initialized")
        index=FAISS.from_documents(chunks,embeddings)
        print("Index created")
        os.makedirs(FAISS_INDEX_PATH,exist_ok=True)
        index.save_local(FAISS_INDEX_PATH)
        print(f"index saved to {FAISS_INDEX_PATH}")
        return index
    except ValueError as ve:
        print(f"config error {ve}")
        return None
    except Exception as e:
        print(f"Error in creating index: {e}")
        return None        

def index_load()-> FAISS:
    try:
        index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(index_file):
            raise FileNotFoundError("Index not found")
        print("loaded index")
        embeddings=create_embeddings()
        if embeddings is None:
            raise ValueError("Embeddings are not initialized")
        index=FAISS.load_local(FAISS_INDEX_PATH,embeddings, allow_dangerous_deserialization=True)
        print("Index loaded")
        return index
    except ValueError as ve:
        print(f"config error {ve}")
        return None
    except Exception as e:
        print(f"Error in loading index: {e}")
        return None
        
def similiarity_search(query: str,index: FAISS)-> tuple[list[Document],list[float]]:
    try:
        if not query:
            raise ValueError("Query is empty")
        if index is None:
            raise ValueError("Index is not initialized")
        from config.config import TOP_K
        results=index.similarity_search_with_score(query,k=TOP_K)
        if not results:
            print("No return from FAISS")
            return [],[]
        chunks, scores=zip(*results)
        max_score=max(scores)
        print(F"return from FAISS with max score: {max_score}")
        return list(chunks),float(max_score)
    except ValueError as ve:
        print(f"config error {ve}")
        return [],[]
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return [],[]

def get_index(csv_path:str=None)-> FAISS:
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(index_file):
        return index_load()
    if csv_path is None:
        raise ValueError("Index not found")
        return None
    from utils.ingest import ingest_data
    chunks=ingest_data(csv_path)
    return create_index(chunks)

if __name__ == "__main__":
    index=get_index("data/drugsComTrain_raw.csv")
    if index:
        test_queries=[
            "What are the side effects of Guanfacine for ADHD?",
            "Is Valsartan effective for heart conditions?",
            "Birth control side effects",
        ]
        for query in test_queries:
            chunks,score=similiarity_search(query,index)
            if not chunks:
                print("No chunks found")
                continue
            else:
                print(f"Top results : {chunks[0].page_content[:100]}")
                print("Metadata:",chunks[0].metadata)
                print("Score:",score)
            