from Ingestor.html_loader import load_html
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
def recursive_splitters(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_texts = splitter.split_text(text)
    documents = []

    for i, chunk_text in enumerate(raw_texts):
        doc = Document(
            page_content=chunk_text,
            metadata={
                "chunk_id": f"chunk_{i:03d}",
                "chunk_index": i,
            }
        )
        documents.append(doc)
    return documents


if __name__ == "__main__":
    doc_data = load_html("https://en.wikipedia.org/wiki/Forest")
    docs = recursive_splitters(doc_data["page_content"])
    print(f"Total chunks created: {len(docs)}")
    print(f"First chunk metadata: {docs[0].metadata}")
    print(f"First chunk length: {len(docs[0].page_content)}")
    