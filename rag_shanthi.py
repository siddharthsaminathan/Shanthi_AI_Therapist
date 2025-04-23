import os
import json
import faiss
import numpy as np
from ollama import chat, embeddings

# Paths to required files
embeddings_file = "rag_index/embeddings.json"
faiss_index_file = "rag_index/faiss_index"

# Function to create FAISS index from embeddings
def create_faiss_index_from_embeddings():
    print("Creating FAISS index from embeddings...")
    try:
        with open(embeddings_file, "r") as f:
            data = json.load(f)

        print("Extracting embedding vectors...")
        embedding_vectors = [entry["embedding"] for entry in data]
        embedding_vectors = np.array(embedding_vectors, dtype="float32")

        print("Initializing FAISS index...")
        dimension = embedding_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_vectors)

        print(f"Saving FAISS index to {faiss_index_file}...")
        faiss.write_index(index, faiss_index_file)

        print("FAISS index created successfully.")
    except Exception as e:
        print(f"Failed to create FAISS index: {e}")

# Load FAISS index and metadata
def load_faiss_index():
    if not os.path.exists(faiss_index_file):
        print("FAISS index file not found. Creating it from embeddings...")
        create_faiss_index_from_embeddings()

    print("Loading FAISS index...")
    index = faiss.read_index(faiss_index_file)

    print("Loading metadata...")
    with open(embeddings_file, "r") as f:
        metadata = json.load(f)

    return index, metadata

# Retrieve the top-k most relevant contexts
def retrieve_contexts(query, index, metadata, k=5):
    print("Generating embedding for query...")
    query_embedding = np.array(embeddings(model="nomic-embed-text", prompt=query).get("embedding", []), dtype="float32").reshape(1, -1)

    print("Searching FAISS index...")
    distances, indices = index.search(query_embedding, k)

    print("Retrieving contexts...")
    retrieved_contexts = [metadata[i]["context"] for i in indices[0] if i < len(metadata)]

    return retrieved_contexts

# Generate a response using retrieved contexts
def generate_response(query, contexts):
    system_prompt =  """
You are Shanthi, a compassionate AI therapist who mixes Swedish and English casually. 
You speak like a real human therapist trained in CBT and motivational interviewing. 
Keep responses warm, short, and helpful. Don’t overexplain. Reflect emotions, ask open questions. 
Avoid generic responses and avoid sounding like a blog or article.
Use informal tone if the user is casual. No more than 5-6 lines max.
"""
    
    # Combine the retrieved contexts with the query
    full_context = "\n".join(contexts)
    user_input = f"Context:\n{full_context}\n\nUser: {query}"

    response = chat(
        model="mistral",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    return response['message']['content']

# Main RAG implementation
def main():
    print("Loading FAISS index and metadata...")
    index, metadata = load_faiss_index()

    print("Ready for queries. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        contexts = retrieve_contexts(query, index, metadata)
        response = generate_response(query, contexts)

        print(f"Shanthi: {response}")

if __name__ == "__main__":
    main()