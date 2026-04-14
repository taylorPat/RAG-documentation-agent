import io
import pathlib
from zipfile import ZipFile
import frontmatter
import requests


def download_zip_file(owner: str, repo: str, branch: str = "main"):
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    response = requests.get(url=url)
    response.raise_for_status()
    return response

def getFilesWithMetadata(content: str,*, extensions: list[str], folder: str | None = None):
    with ZipFile(io.BytesIO(content)) as zfile:
        for file_info in zfile.infolist():
            filename = file_info.filename.lower()
            if folder:
                parts = filename.strip("/").split("/")
                if folder not in parts:
                    continue
            if pathlib.Path(filename).suffix in set(extensions):
                with zfile.open(file_info, mode="r") as file:
                    raw = file.read()
                    try:
                        md_content = raw.decode("utf-8")
                    except Exception:
                        md_content = raw.decode("utf-8", errors="replace")
                    fm_content = frontmatter.loads(md_content)
                    fm_content_dict = fm_content.to_dict()
                    fm_content_dict["content"] = fm_content.content
                    fm_content_dict["filename"] = filename
                    yield fm_content_dict
                    

def get_files_from_repo(owner: str, repo: str, branch: str, extensions: list[str], folder: str | None = None):
    response = download_zip_file(owner=owner, repo=repo, branch=branch)
    yield from getFilesWithMetadata(response.content, extensions=extensions, folder=folder)