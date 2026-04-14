import asyncio
import streamlit as st
import numpy as np
import os
import hashlib

from src.download import get_files_from_repo
from src.FE.components import repo_input_form
from src.chunk import chunk_by_sliding_window
from src.embedding import Embedding
from src.search import SemanticSearch, TextSearch
from src.agent import create_agent


st.set_page_config(page_title="Repo Markdown Explorer", layout="wide")
st.title("📚 Repo Markdown Explorer")


# -------------------------
# HELPERS
# -------------------------

def make_id(owner, repo, branch, folder):
    base = f"{owner}_{repo}_{branch}_{folder}"
    return hashlib.md5(base.encode()).hexdigest()


def get_embedding_path(repo_id):
    os.makedirs("embeddings", exist_ok=True)
    return f"embeddings/{repo_id}.npy"


# -------------------------
# INPUT
# -------------------------

owner, repo, branch, folder, fetch = repo_input_form()
repo_id = make_id(owner, repo, branch, folder)


# -------------------------
# PROCESSING (ONLY ON FETCH)
# -------------------------

if fetch:
    status = st.empty()

    try:
        status.info("Downloading repository...")
        data = list(
            get_files_from_repo(
                owner=owner,
                repo=repo,
                extensions=[".md", ".mdx"],
                branch=branch,
                folder=folder or None,
            )
        )
        st.session_state["data"] = data

        status.info("Chunking documents...")
        chunks = chunk_by_sliding_window(documents=data)
        st.session_state["chunks"] = chunks

        status.info("Handling embeddings...")
        emb_path = get_embedding_path(repo_id)

        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
        else:
            embedding = Embedding()
            contents = [c.get("content", "") for c in chunks]

            try:
                embeddings = np.array(embedding.create(content=contents))
            except Exception:
                embeddings = np.array(
                    [embedding.create(content=c) for c in contents]
                )

            np.save(emb_path, embeddings)

        st.session_state["embeddings"] = embeddings

        status.success(
            f"{len(data)} files | {len(chunks)} chunks | {len(embeddings)} embeddings ready"
        )

    except Exception as e:
        status.error(f"Processing error: {e}")


# -------------------------
# LOAD STATE (PERSISTENT)
# -------------------------

data = st.session_state.get("data", [])
chunks = st.session_state.get("chunks", [])
embeddings = st.session_state.get("embeddings", None)


# -------------------------
# SINGLE STATUS DISPLAY
# -------------------------

if data or chunks:
    st.divider()
    st.subheader("📊 Repository Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Files", len(data))

    with col2:
        st.metric("Chunks", len(chunks))

    with col3:
        st.metric("Embeddings", len(embeddings) if embeddings is not None else 0)


# -------------------------
# QUERY UI (STABLE)
# -------------------------

if chunks:
    st.divider()

    col1, col2, col3, col4 = st.columns([6, 2, 2, 1])

    with col1:
        query = st.text_input("Ask a question", key="query_input")

    with col2:
        model = st.selectbox(
            "Model",
            ["kimi-k2.5:cloud", "qwen2.5:0.5b"],
            index=1,
            key="model_select",
        )

    with col3:
        tool_choice = st.selectbox(
            "Search mode",
            ["None", "Text", "Semantic"],
            index=1,
            key="tool_select",
        )

    with col4:
        ask_clicked = st.button("Ask")


    # -------------------------
    # EXECUTION
    # -------------------------

    if ask_clicked and query:
        with st.spinner("Running..."):
            tools = []
            results = []

            # -------- tools --------
            try:
                if tool_choice == "Text":
                    ts = TextSearch.create_from_chunks(
                        text_fields=["chunk", "title", "description", "filename"],
                        chunks=chunks,
                    )
                    tools.append(ts.search)

                elif tool_choice == "Semantic":
                    if embeddings is None:
                        st.error("Embeddings missing")
                    else:
                        ss = SemanticSearch.create_from_chunks(
                            embedded_chunks=embeddings,
                            chunks=chunks,
                        )
                        tools.append(ss.search)

            except Exception as e:
                st.error(f"Tool error: {e}")

            # -------- search preview --------
            if tools:
                try:
                    results = tools[0](query, num_results=5)
                except TypeError:
                    results = tools[0](query)

            # -------- agent --------
            try:
                agent = create_agent(tools=tools, model_name=model)

                async def run():
                    return await agent.run(query)

                response = asyncio.run(run())
                output = getattr(response, "output", None)

            except Exception as e:
                st.error(f"Agent error: {e}")
                output = None

            # -------- display --------
            if output:
                st.subheader("🧠 Answer")
                st.write(output)
                
            with st.expander("🔍 Agent reasoning results"):
                for m in response.all_messages():
                    st.write(m)

            with st.expander("🔍 Search results"):
                st.write(results)