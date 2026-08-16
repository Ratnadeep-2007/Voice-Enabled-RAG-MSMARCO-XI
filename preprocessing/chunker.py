"""
Chunking strategies for VoiceRAG:
- Strategy A: Fixed-size chunking (no overlap)
- Strategy B: Fixed-size chunking with overlap
- Strategy C: Sentence-aware / semantic splitting
- Strategy D: Adaptive token-length + metadata-aware chunking (Baseline Proposed in PRD §11)
"""

import re
from enum import Enum
from typing import List, Dict, Any, Optional
from .cleaner import TextCleaner

class ChunkingStrategy(str, Enum):
    FIXED = "fixed" # Strategy A
    FIXED_OVERLAP = "fixed_overlap" # Strategy B
    SENTENCE = "sentence" # Strategy C
    ADAPTIVE = "adaptive" # Strategy D

class DocumentChunker:
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        min_chunk_size: int = 64,
        max_chunk_size: int = 384
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _split_into_sentences(self, text: str) -> List[str]:
        # Multilingual sentence splitting supporting Indic danda (।), double danda (॥), ., !, ?
        sentence_pattern = r"(?<=[.!?।॥])\s+"
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_fixed(self, text: str, overlap: bool = False) -> List[str]:
        words = text.split()
        if not words:
            return []
        
        step = self.chunk_size - (self.chunk_overlap if overlap else 0)
        step = max(1, step)
        
        chunks = []
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def chunk_sentence_aware(self, text: str) -> List[str]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text] if text else []
        
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def chunk_adaptive(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Adaptive + metadata-aware chunking (PRD §11 Strategy D):
        1. Sentence boundary detection (multilingual).
        2. Groups sentences respecting min/max token length constraints.
        3. Adds small overlap context where beneficial.
        4. Injects context title and metadata traceability.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        raw_chunks = []
        current_sentences = []
        current_word_count = 0

        for sent in sentences:
            word_count = len(sent.split())
            if current_word_count + word_count > self.max_chunk_size and current_sentences:
                raw_chunks.append(" ".join(current_sentences))
                # Add overlap of last sentence if it helps continuity
                if self.chunk_overlap > 0 and len(current_sentences) > 1:
                    last_sent = current_sentences[-1]
                    current_sentences = [last_sent, sent]
                    current_word_count = len(last_sent.split()) + word_count
                else:
                    current_sentences = [sent]
                    current_word_count = word_count
            else:
                current_sentences.append(sent)
                current_word_count += word_count

        if current_sentences:
            raw_chunks.append(" ".join(current_sentences))

        # Format with metadata
        processed_chunks = []
        doc_id = metadata.get("doc_id", "doc_unknown") if metadata else "doc_unknown"
        doc_title = metadata.get("title", "") if metadata else ""
        lang = metadata.get("language", "en") if metadata else "en"
        source = metadata.get("source", "MSMARCO-XI") if metadata else "MSMARCO-XI"

        for idx, chunk_text in enumerate(raw_chunks):
            # Optional title prepending for contextual embedding quality
            annotated_text = f"[{doc_title}] {chunk_text}" if doc_title else chunk_text
            chunk_dict = {
                "chunk_id": f"{doc_id}_c{idx:03d}",
                "doc_id": doc_id,
                "position": idx,
                "total_chunks": len(raw_chunks),
                "text": chunk_text,
                "annotated_text": annotated_text,
                "token_count": len(chunk_text.split()),
                "language": lang,
                "source": source,
                "title": doc_title,
                "strategy": "adaptive"
            }
            processed_chunks.append(chunk_dict)

        return processed_chunks

    def chunk_document(
        self,
        document: Dict[str, Any],
        strategy: Optional[ChunkingStrategy] = None
    ) -> List[Dict[str, Any]]:
        active_strategy = strategy or self.strategy
        passage = document.get("passage", "")
        cleaned_passage = TextCleaner.normalize_text(passage)
        doc_id = document.get("doc_id", "doc_000")
        doc_title = document.get("title", "")
        lang = document.get("language", "en")
        source = document.get("source", "MSMARCO-XI")

        if active_strategy == ChunkingStrategy.ADAPTIVE:
            return self.chunk_adaptive(cleaned_passage, metadata=document)

        elif active_strategy == ChunkingStrategy.FIXED:
            raw_texts = self.chunk_fixed(cleaned_passage, overlap=False)
        elif active_strategy == ChunkingStrategy.FIXED_OVERLAP:
            raw_texts = self.chunk_fixed(cleaned_passage, overlap=True)
        elif active_strategy == ChunkingStrategy.SENTENCE:
            raw_texts = self.chunk_sentence_aware(cleaned_passage)
        else:
            raw_texts = [cleaned_passage]

        # Standardize output structure
        chunks = []
        for idx, text in enumerate(raw_texts):
            chunks.append({
                "chunk_id": f"{doc_id}_c{idx:03d}",
                "doc_id": doc_id,
                "position": idx,
                "total_chunks": len(raw_texts),
                "text": text,
                "annotated_text": text,
                "token_count": len(text.split()),
                "language": lang,
                "source": source,
                "title": doc_title,
                "strategy": active_strategy.value
            })
        return chunks

    def chunk_corpus(
        self,
        documents: List[Dict[str, Any]],
        strategy: Optional[ChunkingStrategy] = None
    ) -> List[Dict[str, Any]]:
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc, strategy=strategy)
            all_chunks.extend(chunks)
        return all_chunks
