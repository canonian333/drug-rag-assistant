import streamlit as st
from utils.vectorstore import get_or_build_index
from utils.chain import run_rag_pipeline

# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Drug & Medicine Assistant",
    page_icon  = "💊",
    layout     = "centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
    <style>
        /* Modern CSS Variables for easy maintenance */
        :root {
            --primary-green: #2e7d32;
            --primary-blue: #1565c0;
            --primary-orange: #e65100;
            --primary-purple: #6a1b9a;
            --primary-yellow: #f9a825;
        }

        .answer-box {
            background-color : rgba(46, 125, 50, 0.1);
            border-left      : 5px solid var(--primary-green);
            padding          : 1.5rem;
            border-radius    : 12px;
            font-size        : 16px;
            line-height      : 1.6;
            color            : inherit;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            backdrop-filter: blur(8px);
        }
        
        .source-badge-db {
            background-color : rgba(21, 101, 192, 0.15);
            border           : 1px solid var(--primary-blue);
            color            : #4dabf5; /* Explicit light blue for dark mode readability */
            padding          : 5px 15px;
            border-radius    : 20px;
            font-size        : 13px;
            font-weight      : 600;
        }
        
        .source-badge-web {
            background-color : rgba(230, 81, 0, 0.15);
            border           : 1px solid var(--primary-orange);
            color            : #ffb74d; /* Explicit light orange */
            padding          : 5px 15px;
            border-radius    : 20px;
            font-size        : 13px;
            font-weight      : 600;
        }
        
        .mode-badge {
            background-color : rgba(106, 27, 154, 0.15);
            border           : 1px solid var(--primary-purple);
            color            : #ce93d8; /* Explicit light purple */
            padding          : 5px 15px;
            border-radius    : 20px;
            font-size        : 13px;
            font-weight      : 600;
        }

        .metadata-box {
            background-color : rgba(255, 255, 255, 0.05);
            border           : 1px solid rgba(255, 255, 255, 0.1);
            padding          : 1rem;
            border-radius    : 10px;
            font-size        : 14px;
            transition: all 0.2s ease;
        }
        .metadata-box:hover {
            background-color : rgba(255, 255, 255, 0.08);
            border-color : rgba(255, 255, 255, 0.2);
        }

        .warning-box {
            background-color : rgba(249, 168, 37, 0.1);
            border-left      : 5px solid var(--primary-yellow);
            padding          : 1rem 1.25rem;
            border-radius    : 10px;
            font-size        : 14px;
            color            : #fff9c4; /* Light yellow text */
        }
        
        /* Handling Streamlit Light Theme specific overrides */
        @media (prefers-color-scheme: light) {
            .source-badge-db { color: var(--primary-blue); }
            .source-badge-web { color: var(--primary-orange); }
            .mode-badge { color: var(--primary-purple); }
            .warning-box { color: #5d4037; }
            .metadata-box { background-color: #fafafa; border-color: #eeeeee; }
        }
    </style>
""", unsafe_allow_html=True)


# ── Load FAISS Index ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_index():
    with st.spinner("⏳ Loading knowledge base..."):
        index = get_or_build_index(csv_path="data/drugsComTrain_raw.csv")
    return index


# ── Header ────────────────────────────────────────────────────────────────────

st.title("💊 Drug & Medicine Assistant")
st.markdown(
    "Ask questions about drug uses, side effects, dosages, and conditions. "
    "Powered by patient reviews and trusted medical web sources."
)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    # ── Response mode toggle ──────────────────────────────
    st.subheader("Response Mode")
    mode = st.radio(
        label      = "Select mode:",
        options    = ["Concise", "Detailed"],
        index      = 0,
        horizontal = False,
        help       = "Concise: 2-3 sentence summary | Detailed: Full explanation with sections"
    )

    # ── Mode description ──────────────────────────────────
    if mode == "Concise":
        st.info("⚡ **Concise Mode**\nShort, direct answers in 2-3 sentences.")
    else:
        st.info("📖 **Detailed Mode**\nIn-depth answers covering uses, dosage, side effects and warnings.")

    st.divider()

    # ── About ─────────────────────────────────────────────
    st.subheader("ℹ️ About")
    st.markdown("""
    - 🗄️ **Local DB**: Patient drug reviews (UCI Dataset)
    - 🌐 **Web Search**: WebMD, FDA, Mayo Clinic, Drugs.com
    - 🤖 **LLM**: Groq (Llama3-8b)
    - 📐 **Embeddings**: all-MiniLM-L6-v2
    """)

    st.divider()

    # ── Disclaimer ────────────────────────────────────────
    st.warning(
        "⚠️ This tool is for informational purposes only. "
        "Always consult a qualified healthcare professional "
        "before making any medical decisions."
    )


# ── Load Index ────────────────────────────────────────────────────────────────

index = load_index()

if index is None:
    st.error("❌ Failed to load knowledge base. Please check your dataset and try again.")
    st.stop()


# ── Query Input ───────────────────────────────────────────────────────────────

st.subheader("🔍 Ask a Question")

query = st.text_input(
    label       = "Enter your question:",
    placeholder = "e.g. What are the side effects of Metformin for diabetes?",
    max_chars   = 300
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask 💬", use_container_width=True)
with col2:
    clear = st.button("Clear 🗑️", use_container_width=True)

if clear:
    st.rerun()


# ── Process Query ─────────────────────────────────────────────────────────────

if submit:

    # ── Validate input ────────────────────────────────────
    if not query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
        st.stop()

    st.divider()

    # ── Run pipeline ──────────────────────────────────────
    with st.spinner("🔎 Searching knowledge base..."):
        result = run_rag_pipeline(
            query = query,
            index = index,
            mode  = mode
        )

    answer   = result["answer"]
    source   = result["source"]
    metadata = result["metadata"]

    # ── Answer section ────────────────────────────────────
    st.subheader("📋 Answer")

    # Mode badge
    mode_icon = "⚡" if mode == "Concise" else "📖"
    st.markdown(
        f'<span class="mode-badge">{mode_icon} {mode} Mode</span>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Answer box
    st.markdown(
        f'<div class="answer-box">{answer}</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Source badge ──────────────────────────────────────
    if source == "database":
        st.markdown(
            '<span class="source-badge-db">🗄️ Answered from Local Database</span>',
            unsafe_allow_html=True
        )
    elif source == "web":
        st.markdown(
            '<span class="source-badge-web">🌐 Answered from Web Search</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="source-badge-web">⚠️ No Source Found</span>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metadata / References ─────────────────────────────
    if metadata:
        with st.expander("📎 View Sources & References"):
            if source == "database":
                for i, item in enumerate(metadata, 1):
                    st.markdown(f"""
                    <div class="metadata-box">
                        <b>Source {i}</b><br>
                        💊 Drug      : {item.get('drug_name', 'N/A')}<br>
                        🏥 Condition : {item.get('condition', 'N/A')}<br>
                        ⭐ Rating    : {item.get('rating', 'N/A')}/10<br>
                        📊 Score     : {item.get('score', 'N/A')}
                    </div><br>
                    """, unsafe_allow_html=True)

            elif source == "web":
                for i, item in enumerate(metadata, 1):
                    st.markdown(f"""
                    <div class="metadata-box">
                        <b>Source {i}</b><br>
                        📄 Title : {item.get('title', 'N/A')}<br>
                        🔗 URL   : <a href="{item.get('url', '#')}" target="_blank">{item.get('url', 'N/A')}</a>
                    </div><br>
                    """, unsafe_allow_html=True)

    # ── Medical disclaimer ────────────────────────────────
    st.markdown("""
        <div class="warning-box">
            ⚠️ <b>Medical Disclaimer:</b> This information is for educational 
            purposes only and should not replace professional medical advice. 
            Always consult a qualified healthcare provider before making any 
            medical decisions.
        </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<center><sub>Built with LangChain · FAISS · Groq · Tavily · Streamlit</sub></center>",
    unsafe_allow_html=True
)
