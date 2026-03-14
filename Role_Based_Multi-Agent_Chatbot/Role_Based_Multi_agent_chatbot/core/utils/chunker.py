from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, source: str, chunk_size=500, chunk_overlap=50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return [
        {
            "content": chunk,
            "metadata": {"source": source, "chunk_index": idx}
        }
        for idx, chunk in enumerate(chunks)
    ]
