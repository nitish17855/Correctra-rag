from bs4 import BeautifulSoup
import requests


def _table_to_markdown(table_tag) -> str:
    """
    Converts an HTML <table> element into a clean, pipe-separated Markdown table.
    Preserves row and column structure without HTML tag bloat.
    """
    rows = []
    for tr in table_tag.find_all("tr"):
        # Extract text from either header cells (<th>) or data cells (<td>)
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def load_html(url: str) -> dict:
    """
    Fetches an HTML document from a URL, sanitizes unwanted noise tags,
    and extracts both:
      1. 'page_content': A flat, newline-separated string for standard chunkers.
      2. 'metadata.page_elements': A structured list of paragraphs and tables
         for document-aware chunking.
    """
    # 1. Headers to avoid SEC EDGAR / Wikipedia 403 Forbidden errors
    headers = {
        "User-Agent": "MyRAGProject user@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml",
    }

    # 2. Fetch the document over HTTP
    response = requests.get(url=url, headers=headers, timeout=30)
    response.raise_for_status()
    html_content = response.text

    # 3. Parse HTML tree with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # 4. Remove non-visible code, styles, and hidden accounting/XBRL tags
    unwanted_tags = [
        "script",
        "style",
        "noscript",
        "svg",
        "img",
        "iframe",
        "object",
        "ix:header",
        "ix:hidden",
    ]
    for tag in soup.find_all(unwanted_tags):
        tag.decompose()

    # 5. Extract flat text (used by Fixed-Size and Recursive Chunkers)
    clean_text = soup.get_text(separator="\n", strip=True)

    # 6. Extract structured elements in reading order (used by Document-Aware Chunker)
    elements = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "table"]):
        if tag.name == "table":
            table_md = _table_to_markdown(tag)
            if table_md.strip():
                elements.append({"type": "table", "text": table_md})
        else:
            text = tag.get_text(strip=True)
            if text:
                elements.append({"type": "paragraph", "text": text})

    # 7. Return the canonical document contract
    return {
        "page_content": clean_text,
        "metadata": {
            "source": url,
            "doc_type": "html",
            "page_elements": elements,
        },
    }


if __name__ == "__main__":
    sample_url = "https://en.wikipedia.org/wiki/Forest"
    doc = load_html(sample_url)
    print(f"Length of flat text: {len(doc['page_content'])}")
    print(f"Source URL: {doc['metadata']['source']}")
    print(f"Total structured elements: {len(doc['metadata']['page_elements'])}")
    
    # Print the types of the first 5 elements as a quick sanity check
    first_few_types = [el["type"] for el in doc["metadata"]["page_elements"][:5]]
    print(f"First 5 element types: {first_few_types}")
