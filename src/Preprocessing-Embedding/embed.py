import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
CLEANED_DIR = "../../data/cleaned_csv"
OUTPUT_EMB = "../../data/embeddings.npy"
OUTPUT_META = "../../data/embedding_data/embeddings.csv"

#TEST for 10 data points
TEST_MODE = False
TEST_LIMIT = 10

# Load all cleaned CSVs
def load_all_cleaned():
    files = [
        "AskCulinary_cleaned.csv",
        "Baking_cleaned.csv",
        "Cooking_cleaned.csv",
        "FoodScience_cleaned.csv",
        "Recipes_cleaned.csv"
    ]

    dfs = []
    for f in files:
        path = os.path.join(CLEANED_DIR, f)
        print(f"Loading {path}")
        df = pd.read_csv(path)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total posts loaded: {len(full_df)}")
    return full_df

# Generate embeddings
def generate_embeddings(texts):
    print("Loading MiniLM model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = []
    for t in tqdm(texts, desc="Embedding posts"):
        emb = model.encode(t)
        embeddings.append(emb)

    return np.array(embeddings)

# Main
if __name__ == "__main__":
    df = load_all_cleaned()

    if TEST_MODE:
        df = df.head(TEST_LIMIT)
        print("TEST MODE ON — embedding only first {TEST_LIMIT} posts")

    # Use full_text for embeddings
    texts = df["full_text"].fillna("").tolist()

    print("Generating embeddings...")
    emb = generate_embeddings(texts)

    print("Saving embeddings...")
    #np.save(OUTPUT_EMB, emb) - other option to save the large embedding
    df['embedding'] = emb.tolist()

    print("Saving metadata...")
    df.to_csv(OUTPUT_META, index=False)

    print("\nDONE! Embeddings + metadata saved.")