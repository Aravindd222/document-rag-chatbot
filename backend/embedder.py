'''from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List

#Load environment variables from .env file
load_dotenv()
#Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small" #1536-dimension vectors

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    #Embeds chunks using OpenAI's embedding model
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model=EMBEDDING_MODEL
        )
        embeddings.append(response.data[0].embedding)
    return embeddings


def embed_user_query(query: str) -> List[float]:
    #Embeds user query using the OpenAI embedding model
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding '''


from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _model

def embed_chunks(chunks):
    model = get_model()
    return model.encode(
        chunks,
        batch_size=32,
        normalize_embeddings=True
    ).tolist()

def embed_user_query(query):
    model = get_model()
    return model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()