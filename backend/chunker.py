from typing import List, Tuple

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

    return chunks