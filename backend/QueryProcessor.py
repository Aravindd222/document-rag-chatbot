from llm import query_llm_with_context
from embedder import embed_user_query
from vectorstore import search_in_pinecone
from reranker import rerank

def process_user_query(query: str):
    #Embed the user query to create a vector representation
    query_vector = embed_user_query(query)

    #search the vector DB to find relevant chunks related to the usr's question
    matched_chunks = search_in_pinecone(query_vector,top_k=8)

    # Step 2: rerank them using cross-encoder
    best_chunks = rerank(query, matched_chunks, top_n=4) 

    # Join chunks into clean context block
    context = "\n\n---\n\n".join(best_chunks)

    #send the user query and search results(query + context) to LLM for generating response
    generated_response = query_llm_with_context(query, context)
    print(generated_response)

if __name__ == "__main__":
    user_query = "What is the purpose of the Business Conduct Guidelines?"
    process_user_query(user_query)
