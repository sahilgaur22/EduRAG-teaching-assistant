import requests
import os
import json
import pandas as pd

def create_embedding(text):

    response = requests.post(
        "http://localhost:11434/api/embed", 
        json = {
        "model" : "qwen3-embedding:4b",
        "input" : text
    })

    response.raise_for_status()

    embedding = response.json()["embeddings"]
    return embedding

jsons = os.listdir("jsons") # List all the JSONs
all_chunks = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding = "utf-8") as f:
        content = json.load(f)

    print(f"Creating Embeddings for {json_file}")

    embeddings = create_embedding([c['text'] for c in content['chunks']])

    # Attach embeddings to original chunks   
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        all_chunks.append(chunk)

# print(all_chunks)
df = pd.DataFrame.from_records(all_chunks)
print(df)
