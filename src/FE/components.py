import streamlit as st


def repo_input_form():
    """Render owner, repo, branch inputs and a fetch button on one line.

    Returns: (owner, repo, branch, submit_pressed)
    """
    # put input controls in the sidebar so main area stays for status/results
    st.sidebar.title("📚 Repo Markdown Explorer")
    input_mode = st.sidebar.radio(
        "",
        ["GitHub repo", "Upload Markdown files"],
        index=0,
        horizontal=True,
    )

    # ensure session state reflects current mode
    st.session_state["input_mode"] = input_mode

    # default values
    owner = ""
    repo = ""
    branch = ""
    folder = ""
    submit = False

    if input_mode == "GitHub repo":
        with st.sidebar.form("repo_form"):
            owner = st.text_input("Owner / Author", value="fastapi")
            repo = st.text_input("Repository name", value="typer")
            branch = st.text_input("Branch", value="master")
            folder = st.text_input("Optional folder (path inside repo)", value="")

            # semantic search toggle and chunk size (appear before Process)
            use_semantic = st.checkbox(
                "Use semantic search (chunk documents for embeddings)",
                value=st.session_state.get("use_semantic", False),
                key="use_semantic_checkbox",
            )
            st.session_state["use_semantic"] = use_semantic

            if use_semantic:
                chunk_size = st.number_input(
                    "Chunk size (chars)",
                    min_value=100,
                    max_value=20000,
                    value=st.session_state.get("chunk_size", 500),
                    step=100,
                    key="chunk_size_input",
                )
                st.session_state["chunk_size"] = int(chunk_size)

            submit = st.form_submit_button("Process")

    else:
        # upload mode: show uploader and a submit button in the sidebar
        uploaded_files = st.sidebar.file_uploader(
            "Upload markdown files",
            type=["md", "mdx", "markdown"],
            accept_multiple_files=True,
        )
        # show filenames for clarity in the sidebar
        if uploaded_files:
            names = [getattr(f, "name", "uploaded.md") for f in uploaded_files]
            st.sidebar.write("Uploaded: " + ", ".join(names))
        # semantic search toggle and chunk size (appear before Process)
        use_semantic = st.sidebar.checkbox(
            "Use semantic search (chunk documents for embeddings)",
            value=st.session_state.get("use_semantic", False),
            key="use_semantic_checkbox_upload",
        )
        st.session_state["use_semantic"] = use_semantic

        if use_semantic:
            chunk_size = st.sidebar.number_input(
                "Chunk size (chars)",
                min_value=100,
                max_value=20000,
                value=st.session_state.get("chunk_size", 500),
                step=100,
                key="chunk_size_input_upload",
            )
            st.session_state["chunk_size"] = int(chunk_size)

        # Use a regular sidebar button so uploader is reactive
        if st.sidebar.button("Process"):
            submit = True
            st.session_state["uploaded_markdown_files"] = uploaded_files or []

    return owner.strip(), repo.strip(), branch.strip(), folder.strip(), submit
