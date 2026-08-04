import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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
# print(df)

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] # Since create_embedding() returns a list of embeddings
# print(question_embedding)

# Find Similarity of question_embedding with Other Embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)

top_results = int(input("Enter the top similarities you want to Fetch: "))
max_idx = similarities.argsort()[ : : -1][0 : top_results] # Reverse sort and get the top indices
# print(max_idx)

new_df = df.loc[max_idx]
print(new_df[["title", "number", "text"]])