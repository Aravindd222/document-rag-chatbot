from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_pinecone
from typing import List


pdf_path = "./resources/IBM_Business_Conduct_Guidelines.pdf"
def run():
    #Read PDF and extract text
    pages = read_pdf(pdf_path)

    #Chunk data into smaller pieces
    chunks = chunk_pages(pages)

    #Embed chunks using OpenAI's embedding model to create vector representations
    embeddings = embed_chunks(chunks)

    #store the chunks and embeddings in Pinecone for efficient retrieval
    store_in_pinecone(chunks, embeddings, namespace="")

if __name__ == "__main__":
    run()

