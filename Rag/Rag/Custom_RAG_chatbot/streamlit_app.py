from pathlib import Path

import ollama
import streamlit as st


DATASET_PATH = Path("cat-facts.txt")
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"


@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: str) -> list[str]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@st.cache_resource(show_spinner=False)
def build_vector_database(chunks: tuple[str, ...]) -> list[tuple[str, list[float]]]:
    vector_db = []
    progress = st.progress(0, text="Building knowledge base...")

    for index, chunk in enumerate(chunks, start=1):
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)["embeddings"][0]
        vector_db.append((chunk, embedding))
        progress.progress(index / len(chunks), text=f"Embedding item {index}/{len(chunks)}")

    progress.empty()
    return vector_db


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(x**2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def retrieve(query: str, vector_db: list[tuple[str, list[float]]], top_n: int) -> list[tuple[str, float]]:
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    similarities = [
        (chunk, cosine_similarity(query_embedding, embedding))
        for chunk, embedding in vector_db
    ]
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_n]


def build_system_prompt(retrieved_knowledge: list[tuple[str, float]]) -> str:
    context = "\n".join(f" - {chunk}" for chunk, _ in retrieved_knowledge)
    return f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Do not make up any new information:
{context}
"""


def stream_answer(prompt: str, retrieved_knowledge: list[tuple[str, float]]):
    system_prompt = build_system_prompt(retrieved_knowledge)
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    for chunk in stream:
        yield chunk["message"]["content"]


st.set_page_config(page_title="Ollama RAG Chatbot", page_icon=":speech_balloon:", layout="centered")

st.title("Ollama RAG Chatbot")
st.caption("Ask questions using the local knowledge in cat-facts.txt.")

with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Retrieved facts", min_value=1, max_value=5, value=3)
    st.write("Embedding model")
    st.code(EMBEDDING_MODEL, language=None)
    st.write("Chat model")
    st.code(LANGUAGE_MODEL, language=None)

try:
    dataset = load_dataset(str(DATASET_PATH))
    if not dataset:
        st.error("cat-facts.txt is empty. Add one fact per line and restart the app.")
        st.stop()

    vector_database = build_vector_database(tuple(dataset))
except Exception as exc:
    st.error(f"Could not prepare the chatbot: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me something about the facts in the local dataset.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context..."):
            retrieved = retrieve(prompt, vector_database, top_n)

        response = st.write_stream(stream_answer(prompt, retrieved))

        with st.expander("Retrieved context"):
            for chunk, similarity in retrieved:
                st.write(f"{similarity:.2f} - {chunk}")

    st.session_state.messages.append({"role": "assistant", "content": response})
