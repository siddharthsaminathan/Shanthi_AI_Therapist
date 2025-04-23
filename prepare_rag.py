import os
import json
from ollama import embeddings
from tqdm import tqdm

# Directory containing preprocessed data
preprocessed_data_dir = "preprocessed_data"
output_dir = "rag_index"
os.makedirs(output_dir, exist_ok=True)

# Path to the preprocessed data file
preprocessed_file = os.path.join(preprocessed_data_dir, "mental_health_counseling_preprocessed.json")

# Path to save the embeddings
embeddings_file = os.path.join(output_dir, "embeddings.json")

# Function to generate embeddings using Ollama Nordic-Embed-Text
def generate_embeddings(text):
    try:
        result = embeddings(model="nomic-embed-text", prompt=text)
        return result
    except Exception as e:
        print(f"Failed to generate embedding for text: {text[:50]}... Error: {e}")
        return None

# Prepare RAG index
def prepare_rag_index():
    try:
        print(f"Loading preprocessed data from {preprocessed_file}...")
        with open(preprocessed_file, "r") as f:
            data = json.load(f)

        embeddings = []
        print("Generating embeddings for each context...")
        for entry in tqdm(data):
            context = entry.get("context", "")
            response = entry.get("response", "")
            embedding_response = generate_embeddings(context)
            if embedding_response:
                # Extract the embedding vector from the response
                embedding_vector = embedding_response.get("embedding", [])
                embeddings.append({
                    "context": context,
                    "response": response,
                    "embedding": embedding_vector
                })

        print(f"Saving embeddings to {embeddings_file}...")
        with open(embeddings_file, "w") as f:
            json.dump(embeddings, f, indent=4)

        print("RAG index preparation complete.")
    except Exception as e:
        print(f"Failed to prepare RAG index: {e}")

if __name__ == "__main__":
    print("Starting RAG index preparation...")
    prepare_rag_index()
    print("RAG index preparation finished.")