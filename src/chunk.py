import re


def chunk_by_sliding_window(documents: list[dict]) -> list[dict]:
    all_chunks = []
    
    for doc in documents:
        doc_copy = doc.copy()
        content = doc_copy.pop("content")
        chunks = _chunk_by_sliding_window(seq=content, chunk_size=2000, step=1000)
        for chunk in chunks:
            chunk.update(doc_copy) # Adds the remaining data from doc_copy to chunk (title', 'description', 'filename')
        all_chunks.extend(chunks)
    return all_chunks


def _chunk_by_sliding_window(seq: list, chunk_size: int, step: int) -> list[dict]:
    if chunk_size <= 0 or step <= 0:
        raise ValueError("Must be greater than 0!")
    result = []
    for i in range(0, len(seq), step):
        chunk_content = seq[i:i+chunk_size]
        result.append({"start": i, "content": chunk_content})
        if i + chunk_size > len(seq):
            break
    return result


def chunk_by_md_section(documents: list[dict]) -> list[dict[str, any]]:
    all_chunks = []

    for doc in documents:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        sections = _chunk_by_md_section(doc_content, level=2)
        for section in sections:
            section_doc = doc_copy.copy()
            section_doc['start'] = section["start"]
            section_doc["content"] = section["content"]
            all_chunks.append(section_doc)
    return all_chunks

def _chunk_by_md_section(content: str, level: int = 1) -> list[dict[str, str]]:
    """
    Split markdown text into chunks based on a specific heading level.

    Each chunk includes:
      - 'start': starting character index of the chunk
      - 'chunk': the full text of the chunk (including heading)

    Args:
        seq (str): Markdown content
        level (int): Heading level (1 = '#', 2 = '##', ...)

    Returns:
        List[Dict]: [{'start': int, 'chunk': str}, ...]
    """
    if level < 1:
        raise ValueError("Heading level must be >= 1")

    pattern = re.compile( r'^(#{' + str(level) + r'} )(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(content))

    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        chunk_content = content[start:end].strip()

        chunks.append({
            "start": start,
            "content": chunk_content
        })

    return chunks