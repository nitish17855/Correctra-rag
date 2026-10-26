import sys
from pathlib import Path
from typing import Optional
import nltk
from langchain_core.documents import Document

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reranker.cross_encoder_reranker import CrossEncoderReranker


class KnowledgeRefiner:
    """
    Performs sentence-level Knowledge Refinement (Decompose -> Score -> Recompose)
    on retrieved chunks using a local cross-encoder to eliminate intra-chunk noise.
    """

    def __init__(
        self,
        reranker: Optional[CrossEncoderReranker] = None,
        sentences_per_strip: int = 2,
        top_k_strips: int = 5,
    ):
        self._reranker = reranker
        self.sentences_per_strip = sentences_per_strip
        self.top_k_strips = top_k_strips

        # Ensure sentence tokenizer is ready
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def decompose(self, documents: list[Document]) -> list[Document]:
        """
        Decomposes accepted documents into smaller evidence strips.

        Rules:
        - Segments text into sentences using NLTK sent_tokenize.
        - If a document has <= sentences_per_strip sentences, keeps it intact.
        - Otherwise, groups adjacent sentences into strips of target size using buffer flushing.
        - Preserves parent chunk ID, original strip index, and document metadata for citations.
        """
        strips: list[Document] = []

        for doc_idx, doc in enumerate(documents):
            text = doc.page_content.strip()
            if not text:
                continue

            parent_id = doc.metadata.get("chunk_id", f"chunk_{doc_idx:03d}")
            sentences = nltk.sent_tokenize(text)

            if not sentences:
                continue

            # Short chunk: keep intact as a single strip
            if len(sentences) <= self.sentences_per_strip:
                meta = dict(doc.metadata)
                meta.update({
                    "parent_chunk_id": parent_id,
                    "strip_index": 0,
                    "sentence_count": len(sentences),
                })
                strips.append(Document(page_content=text, metadata=meta))
                continue

            # Longer chunk: group adjacent sentences into strips
            current_buffer: list[str] = []
            strip_idx = 0

            def _flush_buffer():
                nonlocal current_buffer, strip_idx
                if current_buffer:
                    strip_text = " ".join(current_buffer)
                    meta = dict(doc.metadata)
                    meta.update({
                        "parent_chunk_id": parent_id,
                        "strip_index": strip_idx,
                        "sentence_count": len(current_buffer),
                    })
                    strips.append(Document(page_content=strip_text, metadata=meta))
                    strip_idx += 1
                    current_buffer = []

            for sent in sentences:
                current_buffer.append(sent)
                if len(current_buffer) >= self.sentences_per_strip:
                    _flush_buffer()

            # Flush any trailing remaining sentences
            _flush_buffer()

        return strips

    def refine(self, query: str, strips: list[Document]) -> list[Document]:
        """
        Scores each strip against the query using joint cross-attention and
        returns the top-k most relevant strips.
        """
        if not strips:
            return []

        # 1. Batch cross-encoder scoring
        pairs = [[query, strip.page_content] for strip in strips]
        scores = self.reranker.model.predict(pairs)

        # 2. Attach scores to strip metadata
        scored_strips: list[Document] = []
        for strip, score in zip(strips, scores):
            meta = dict(strip.metadata)
            meta["refine_score"] = round(float(score), 4)
            scored_strips.append(
                Document(page_content=strip.page_content, metadata=meta)
            )

        # 3. Sort descending by relevance score and take top-k
        scored_strips.sort(
            key=lambda s: s.metadata["refine_score"], reverse=True
        )
        return scored_strips[: self.top_k_strips]

    def recompose(self, retained_strips: list[Document]) -> list[Document]:
        """
        Recomposes retained strips back into cohesive parent document chunks.
        Sorts strips by their original strip_index so sentence order is preserved.
        """
        if not retained_strips:
            return []

        from collections import defaultdict
        grouped = defaultdict(list)
        for strip in retained_strips:
            parent_id = strip.metadata.get("parent_chunk_id", "unknown")
            grouped[parent_id].append(strip)

        recomposed_docs: list[Document] = []
        for parent_id, strips in grouped.items():
            # Crucial: restore source narrative order by strip_index
            strips.sort(key=lambda s: s.metadata.get("strip_index", 0))

            recomposed_text = " ".join(s.page_content for s in strips)
            
            # Base metadata from the first strip, adding refinement audit info
            base_meta = dict(strips[0].metadata)
            base_meta["chunk_id"] = parent_id
            base_meta["retained_strips"] = [s.metadata.get("strip_index") for s in strips]
            base_meta["max_refine_score"] = max(s.metadata.get("refine_score", 0.0) for s in strips)

            recomposed_docs.append(
                Document(page_content=recomposed_text, metadata=base_meta)
            )

        return recomposed_docs

    def filter_and_recompose(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        """
        End-to-end Knowledge Refinement:
        Decompose chunks -> Score and Filter Strips -> Recompose surviving evidence.
        """
        if not documents:
            return []

        strips = self.decompose(documents)
        if not strips:
            return []

        refined_strips = self.refine(query=query, strips=strips)
        return self.recompose(refined_strips)
