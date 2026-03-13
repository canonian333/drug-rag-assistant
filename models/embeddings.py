from langchain_huggingface import HuggingFaceEmbeddings
from config.config import EMBEDDING_PROVIDER,EMBEDDING_MODEL_NAME

def create_embeddings() -> HuggingFaceEmbeddings:
   try:
    print(f"Loading Embedding model : {EMBEDDING_MODEL_NAME}")
    embeddings=HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs= {"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Embedding model loaded{EMBEDDING_MODEL_NAME}")
    return embeddings
   except Exception as e:
    print(f"Error loading embedding model: {e}")
    return None

def validate_embeddings(embeddings)-> bool:
    try:
        if embeddings is None:
            raise ValueError("Embeddings are not initialized")
        test_embedding=embeddings.embed_query("test")
        if len(test_embedding) == 0:
            raise ValueError("Embeddings are not initialized")
        print("Embedding validated successfully")
    except ValueError as ve:
        print(f"Error validating embeddings: {ve}")
        return False
    except Exception as e:
        print(f"Error validating embeddings: {e}")
        return False
if __name__ == "__main__":
    embeddings=create_embeddings()
    validate_embeddings(embeddings)
