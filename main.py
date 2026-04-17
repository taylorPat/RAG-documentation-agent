import asyncio
import streamlit as st
import numpy as np
import os
import hashlib
import frontmatter

from src.download import get_files_from_repo
from src.FE.components import repo_input_form
from src.chunk import chunk_by_sliding_window
from src.embedding import Embedding
from src.search import SemanticSearch, TextSearch
from src.agent import create_agent


st.set_page_config(page_title="Repo Markdown Explorer", layout="wide")


# -------------------------
# HELPERS
# -------------------------

def make_id(owner, repo, branch, folder):
    base = f"{owner}_{repo}_{branch}_{folder}"
    return hashlib.md5(base.encode()).hexdigest()


def get_embedding_path(repo_id):
    return f"embeddings/{repo_id}.npy"


# -------------------------
# INPUT
# -------------------------

owner, repo, branch, folder, fetch = repo_input_form()

# compute repo_id: for uploads create a unique id to avoid reusing stale embeddings
if st.session_state.get("input_mode") == "Upload Markdown files":
    import time
    uploaded = st.session_state.get("uploaded_markdown_files", [])
    names = [getattr(f, "name", "uploaded.md") for f in uploaded]
    base = "upload_" + ",".join(names) + "_" + str(time.time())
    repo_id = hashlib.md5(base.encode()).hexdigest()
else:
    repo_id = make_id(owner, repo, branch, folder)


# -------------------------
# PROCESSING (ONLY ON FETCH)
# -------------------------

if fetch:
    try:
        input_mode = st.session_state.get("input_mode")

        # DOWNLOAD (only for GitHub mode) — show spinner in sidebar
        if input_mode == "GitHub repo":
            with st.sidebar:
                with st.spinner("Downloading repository..."):
                    data = list(
                        get_files_from_repo(
                            owner=owner,
                            repo=repo,
                            extensions=[".md", ".mdx"],
                            branch=branch,
                            folder=folder or None,
                        )
                    )
        else:
            # UPLOAD MODE — no downloading message in main; just read uploaded files
            uploaded = st.session_state.get("uploaded_markdown_files", [])
            data = []
            for f in uploaded:
                try:
                    raw = f.read()
                    try:
                        md_content = raw.decode("utf-8")
                    except Exception:
                        md_content = raw.decode("utf-8", errors="replace")
                    fm = frontmatter.loads(md_content)
                    fm_dict = fm.to_dict()
                    fm_dict["content"] = fm.content
                    fm_dict["filename"] = getattr(f, "name", "uploaded.md")
                    data.append(fm_dict)
                except Exception:
                    try:
                        text = raw.decode("utf-8", errors="replace")
                    except Exception:
                        text = ""
                    data.append({"content": text, "filename": getattr(f, "name", "uploaded.md")})

        st.session_state["data"] = data

        # chunking: show spinner in sidebar only when semantic search is enabled
        use_semantic = st.session_state.get("use_semantic", False)
        if use_semantic:
            chunk_size = st.session_state.get("chunk_size", 2000)
            with st.sidebar:
                with st.spinner("Chunking documents..."):
                    chunks = chunk_by_sliding_window(
                        documents=data, chunk_size=chunk_size, step=chunk_size
                    )
        else:
            chunks = chunk_by_sliding_window(documents=data)

        st.session_state["chunks"] = chunks

        # embeddings: only when semantic search enabled
        if use_semantic:
            emb_path = get_embedding_path(repo_id)
            st.session_state["processing_embeddings"] = True
            try:
                with st.sidebar:
                    with st.spinner("Handling embeddings..."):
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
            finally:
                st.session_state["processing_embeddings"] = False

            st.session_state["embeddings"] = embeddings
        else:
            st.session_state["embeddings"] = None

    except Exception as e:
        st.sidebar.error(f"Processing error: {e}")


# -------------------------
# LOAD STATE (PERSISTENT)
# -------------------------

data = st.session_state.get("data", [])
chunks = st.session_state.get("chunks", [])
embeddings = st.session_state.get("embeddings", None)


# -------------------------
# SIDEBAR STATUS (show only after processing)
# -------------------------
if data or chunks:
    # hide metrics while embeddings are being processed
    if not st.session_state.get("processing_embeddings", False):
        with st.sidebar:
            st.divider()
            st.subheader("📊 Repository Status")
            col1, col2, col3 = st.columns(3)
            col1.metric("Files", len(data))
            col2.metric("Chunks", len(chunks))
            # show embeddings only when semantic search is enabled
            if st.session_state.get("use_semantic", False) and embeddings is not None:
                col3.metric("Embeddings", len(embeddings))


# NOTE: repository status moved to sidebar only


# -------------------------
# QUERY UI (STABLE)
# -------------------------

if chunks:
    st.markdown("## Ask a question")

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
        # limit search mode options unless semantic search was enabled in sidebar
        use_semantic = st.session_state.get("use_semantic", False)
        mode_options = ["None", "Text"] + (["Semantic"] if use_semantic else [])
        default_index = 1 if "Text" in mode_options else 0
        tool_choice = st.selectbox(
            "Search mode",
            mode_options,
            index=default_index,
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