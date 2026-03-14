# Drug & Medicine Assistant 💊

A smart RAG-based assistant that answers patient questions about drugs using:
- 🗄️ **Local DB**: Patient reviews from UCI Dataset
- 🌐 **Web Search**: Trusted sources like Drugs.com
- 🧠 **LLM**: Groq Llama 3.3 70B
- 🎯 **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)

## Features
- **Dual-Mode Responses**: Concise (2-3 sentences) or Detailed (in-depth explanation)
- **Smart Retrieval**: Hybrid search combining FAISS similarity + keyword matching
- **Web Fallback**: Automatically searches the web when DB lacks sufficient context
- **Safety First**: Built-in disclaimers and no medical advice

## Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key (set `GROQ_API_KEY` in `.env`)
- Tavily API Key (set `TAVILY_API_KEY` in `.env`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/canonian333/drug-rag-assistant.git
   cd drug-rag-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## How It Works

1. **Load Data**: Loads patient reviews from `data/drugsComTrain_raw.csv`
2. **Build Index**: Creates a FAISS index with sentence embeddings
3. **User Query**: User asks a question (e.g., "What are the side effects of NuvaRing?")
4. **Smart Retrieval**:
   - Checks FAISS index for similar reviews (threshold: 0.7)
   - If insufficient context, performs Tavily web search
   - Combines DB + web results
5. **Generate Answer**:
   - Groq Llama 3.3 70B generates response using retrieved context
   - Supports Concise or Detailed mode
6. **Display**: Shows answer with source attribution and safety warnings

## Project Structure

```
drug-rag-assistant/
├── app.py                # Streamlit application
├── .env                  # Environment variables (not in git)
├── requirements.txt      # Python dependencies
├── config/
│   ├── config.py         # Configuration and constants
│   └── prompts.py        # Prompt templates
├── data/
│   └── drugsComTrain_raw.csv  # Patient reviews dataset
├── models/
│   ├── embedding.py      # Embedding model loading
│   └── llm.py            # LLM initialization
├── utils/
│   ├── chain.py          # RAG pipeline and answer generation
│   ├── retriever.py      # Smart retrieval logic
│   └── vectorstore.py    # FAISS index management
└── README.md             # Project documentation
```

## Testing

Run the built-in test cases:
```bash
python utils/chain.py
```

## License

MIT License