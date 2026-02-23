'''from typing import List, Tuple

def chunk_pages(pages: List[str],chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    chunks: List[str] = []

    full_text = " ".join(pages)
    text_length = len(full_text)

    if(text_length == 0):
        return chunks
    
    start = 0
    while start < text_length:
        #calculate the end index
        end = min(start + chunk_size, text_length)

        #extract the chunk
        chunk = full_text[start:end].strip()
        if chunk: #only add non-empty chunks
            chunks.append(chunk)

        #If this was the last chunk break the loop
        if end >= text_length:
            break

        #calculate next starting position
        start = end - chunk_overlap
        print("starting new chunk at index:", start)

    return chunks'''

from typing import List
import re

# ---- Simple token estimator (fast, no tokenizer dependency) ----
# Rule of thumb: 1 token ≈ 4 characters in English
def estimate_tokens(text: str) -> int:
    return len(text) // 4


def split_into_sentences(text: str) -> List[str]:
    """
    Sentence-aware splitting using regex.
    Keeps punctuation attached to the sentence.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages(
    pages: List[str],
    max_tokens: int = 350,      # target chunk size for embedding
    overlap_sentences: int = 2  # semantic overlap
) -> List[str]:

    chunks: List[str] = []

    full_text = " ".join(pages)
    if not full_text.strip():
        return chunks

    sentences = split_into_sentences(full_text)

    current_chunk: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        # If adding this sentence exceeds budget → finalize chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # ---- Create semantic overlap ----
            overlap = current_chunk[-overlap_sentences:]
            current_chunk = overlap.copy()
            current_tokens = sum(estimate_tokens(s) for s in current_chunk)

        # Add sentence to chunk
        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"Created {len(chunks)} semantic chunks.")
    return chunks