import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_groq import ChatGroq
from config.config import (
LLM_PROVIDER,
GROQ_API_KEY,
GROQ_MODEL,
CONCISE_MAX_TOKENS,
DETAILED_MAX_TOKENS
 )

# setting the default response as detailed
def get_llm(mode:str="concise"):
    if mode=="concise":
        max_tokens=CONCISE_MAX_TOKENS
    elif mode=="detailed":
        max_tokens=DETAILED_MAX_TOKENS
    else:
        max_tokens=DETAILED_MAX_TOKENS

    try:
        if not GROQ_API_KEY:
            raise ValueError("API Key is missing")
        llm=ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            temperature=0.3
        )
        print(f"LLM loaded succesfully: {GROQ_MODEL} , Mode: {mode}, Max_tokens: {max_tokens} ")
        return llm
    except ValueError as ve:
        print(f"Config error {ve}")
        return None
    except Exception as e:
        print(f"LLM Initialization error {e}")
        return None

def llm_validation(llm) -> bool:
    try:
        if llm is None:
            raise ValueError("LLM Instance is not running")
        test_response =llm.invoke("Ok")
        if test_response and test_response.content:
            print(f"LLM instance is running : {test_response.content.strip()}")
            return True
        else:
            raise ValueError("Empty LLM response")
    except ValueError as ve:
        print(f"validation error {ve}")
        return False
    except Exception as e:
        print(f"LLM validation failed {e}")
        return False
if __name__=="__main__":
    #tesing concise_mode
    llm_concise = get_llm(mode="concise")
    llm_validation(llm_concise)

    # #Testing detailed_mode
    llm_detailed = get_llm(mode="detailed")
    llm_validation(llm_detailed)


