import streamlit as st


def repo_input_form():
    """Render owner, repo, branch inputs and a fetch button on one line.

    Returns: (owner, repo, branch, submit_pressed)
    """
    with st.form("repo_form"):
        c1, c2, c3, c4, c5 = st.columns([2, 3, 2, 1, 2])
        with c1:
            owner = st.text_input("Owner / Author", value="fastapi")
        with c2:
            repo = st.text_input("Repository name", value="typer")
        with c3:
            branch = st.text_input("Branch", value="master")
        with c5:
            folder = st.text_input("Optional folder (path inside repo)", value="")
        with c4:
            submit = st.form_submit_button("Fetch")
    return owner.strip(), repo.strip(), branch.strip(), folder.strip(), submit
