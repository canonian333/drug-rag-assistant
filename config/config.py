import os 
from dotenv import load_dotenv

load_dotenv()

#API Keys
GROQ_API_KEY=os.getenv("GROQ_API")
TRAVILY_API=os.getenv("TRAVILY_API")

#LLM Provider
LLM_PROVIDER = "groq"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding model(hugging_face_all-MiniLM-L6-v2)
EMBEDDING_PROVIDER = "huggingface"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

#VectorDB
FAISS_INDEX_PATH = "faiss_index/"

#chunking_strategy
CHUNKING_SIZE = 500
CHUNKING_OVERLAP = 50

#Travely_config(maximum 2 webpages to refer and retrieve)
TRAVELY_MAX_RESULTS= 2

#Generation_logic(concise & detailed by hybrid approach)
CONCISE_MAX_TOKENS    = 150
DETAILED_MAX_TOKENS   = 800
CONCISE_INSTRUCTION   = "Answer in 2-3 sentences only. Be direct and avoid elaboration."
DETAILED_INSTRUCTION  = "Provide a thorough explanation covering drug uses, dosage, side effects, and any warnings."
