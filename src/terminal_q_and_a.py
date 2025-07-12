"""
Terminal-based RAG using Groq LLaMA + Chroma + LangChain
To run:
    python src/terminal_q_and_a.py
"""
import os
import yaml
from typing import List, Tuple
from openai import OpenAI

# LangChain imports (v0.2+)
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from utils.load_config import LoadConfig

# Load app configuration and set env vars
APPCFG = LoadConfig()

# Choose embedding model
if APPCFG.embedding_model_engine.startswith("text-embedding"):
    embedding = OpenAIEmbeddings()
else:
    embedding = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

# Load vector database
vectordb = Chroma(
    persist_directory=APPCFG.persist_directory,
    embedding_function=embedding
)

print("Loaded vector store.")
print("Number of vectors in vectordb:", vectordb._collection.count())

# Initialize OpenAI client with Groq config
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
)

# Terminal-based QA loop
while True:
    question = input("\n\nEnter your question or press 'q' to quit: ")
    if question.lower() == 'q':
        break

    # Retrieve relevant chunks
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    retrieved_content = "\n\n".join(doc.page_content for doc in docs)

    # RAG-style prompt
    prompt = f"""# Retrieved content:\n{retrieved_content}

# User question:\n{question}"""

    # LLM call (Groq)
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "llama3-8b-8192"),
            messages=[
                {"role": "system", "content": APPCFG.llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=APPCFG.temperature,
            max_tokens=APPCFG.max_token
        )

        print("\nResponse:\n", response.choices[0].message.content)

    except Exception as e:
        print("Error during LLM call:", str(e))
