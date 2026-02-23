from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_llm_with_context(query: str, context: str):

    prompt = f"""
You are a retrieval-based assistant.
You are answering questions based on retrieved documents.

Use ALL relevant context to form a complete explanation.
Summarize the key idea instead of listing every section.
Do not add information outside the context.
If the answer is not in the context, say you cannot find it.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message.content