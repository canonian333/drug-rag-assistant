import html
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import (
    CHUNKING_SIZE, 
    CHUNKING_OVERLAP)

def load_clean_data(csv_path: str)-> pd.DataFrame:
    try:
        df=pd.read_csv(csv_path)
        print(f"dataset loaded : {csv_path}")
        df=df.drop(columns=["date","usefulCount"],axis=1, errors="ignore")
        df=df.dropna(subset=["drugName","condition","review","rating"]) 
        #remove duplicates
        df=df.drop_duplicates(subset=["drugName","condition","review"])
        #clean reviews(strip,unescape,replace,duplicates)
        df["review"]=df["review"].apply(lambda x: html.unescape(x))
        df["review"]=df["review"].apply(lambda x: x.strip())
        df["drugName"]=df["drugName"].apply(lambda x: x.strip())
        df["condition"]=df["condition"].apply(lambda x: x.strip())
        df["review"]=df["review"].str.replace("\n","",regex=False)
        df["review"]=df["review"].str.replace("\t","",regex=False)
        df["drugName"]=df["drugName"].str.replace("\t","",regex=False)
        df["condition"]=df["condition"].str.replace("\t","",regex=False)
        df=df.drop_duplicates(subset=["review"])
        print(f"cleaned dataset : {len(df)}")
        df=df.reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Error loading dataset: {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def format_content(row:pd.Series)-> str:
    return f"""
        Drug: {row['drugName']}
        Condition: {row['condition']}
        Review: {row['review']}"""
    
def build_documents(df: pd.DataFrame) -> list[Document]:
    try:
        documents = []
        for _, row in df.iterrows():
            context_header = f"Drug: {row['drugName']} | Condition: {row['condition']}"
            review_content = row['review']
            full_content = context_header + review_content
            metadata = {
                "drug_name": row["drugName"],
                "condition": row["condition"],
                "rating": int(row["rating"])
            }
            documents.append(Document(page_content=full_content, metadata=metadata))
        print(f"Documents built: {len(documents)}")
        return documents
    except Exception as e:
        print(f"Error building documents: {e}")
        return []

#chunking of the documents(recursive)
def chunking(documents:list[Document])-> list[Document]:
    try:
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=CHUNKING_SIZE,
            chunk_overlap=CHUNKING_OVERLAP,
            separators=["\n\n","\n",""],
            length_function=len,  
        )
        chunks=text_splitter.split_documents(documents)
        print(f"Documents chunked: {len(chunks)}")
        return chunks
    except Exception as e:
        print(f"Error chunking documents: {e}")
        return []    
def ingest_data(csv_path: str)-> list[Document]:
    df=load_clean_data(csv_path)
    if df is None or df.empty:
        print("No data to ingest")
        return []
    documents=build_documents(df)
    if not documents:
        print("failed to ingest data")
        return []
    chunks=chunking(documents)
    if not chunks:
        print("failed to chunk data")
        return []
    print(f"Ingestion completed: {len(chunks)} chunks")
    print(F"meta data :{chunks[0].metadata}")
    return chunks
    
if __name__ == "__main__":
    chunks = ingest_data("data/drugsComTrain_raw.csv")
        
        

