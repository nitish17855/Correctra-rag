from langchain_core.documents import Document
from Ingestor.html_loader import load_html


def document_aware_chunker(
    elements: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Document-Aware Chunker:
    Iterates through structured elements (paragraphs, headings, tables) in document order.
    - Treats tables as atomic units (gives them dedicated chunks with is_table=True).
    - Accumulates narrative paragraphs into chunks up to chunk_size without splitting tables.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Document] = []
    current_buffer: list[str] = []
    current_length: int = 0
    chunk_index: int = 0

    def _flush_buffer():
        nonlocal current_buffer, current_length, chunk_index
        if current_buffer:
            chunks.append(
                Document(
                    page_content="\n\n".join(current_buffer),
                    metadata={
                        "chunk_id": f"chunk_{chunk_index:03d}",
                        "chunk_index": chunk_index,
                        "is_table": False,
                        "chunk_strategy": "document_aware",
                    },
                )
            )
            chunk_index += 1
            current_buffer = []
            current_length = 0

    for element in elements:
        el_type = element.get("type", "paragraph")
        el_text = element.get("text", "").strip()

        if not el_text:
            continue

        if el_type == "table":
            # 1. Flush any pending paragraphs before the table
            _flush_buffer()

            # 2. Package the table as its own dedicated chunk
            chunks.append(
                Document(
                    page_content=el_text,
                    metadata={
                        "chunk_id": f"chunk_{chunk_index:03d}",
                        "chunk_index": chunk_index,
                        "is_table": True,
                        "chunk_strategy": "document_aware",
                    },
                )
            )
            chunk_index += 1

        else:
            # 3. For regular paragraphs: check if adding this paragraph exceeds chunk_size
            if current_length + len(el_text) > chunk_size and current_buffer:
                _flush_buffer()

            current_buffer.append(el_text)
            current_length += len(el_text)

    # 4. Flush any remaining paragraphs at the end of the document
    _flush_buffer()

    return chunks


if __name__ == "__main__":
    sample_url = "https://en.wikipedia.org/wiki/Forest"
    doc_data = load_html(sample_url)
    elements = doc_data["metadata"].get("page_elements", [])

    chunked_docs = document_aware_chunker(elements=elements, chunk_size=1000)

    table_chunks = [c for c in chunked_docs if c.metadata.get("is_table")]
    text_chunks = [c for c in chunked_docs if not c.metadata.get("is_table")]

    print(f"Total structured elements ingested: {len(elements)}")
    print(f"Total document-aware chunks created: {len(chunked_docs)}")
    print(f"  - Table chunks: {len(table_chunks)}")
    print(f"  - Text chunks:  {len(text_chunks)}")
    print(f"\nFirst chunk metadata: {chunked_docs[0].metadata}")
    if table_chunks:
        print(f"First table chunk ID: {table_chunks[0].metadata['chunk_id']}")
