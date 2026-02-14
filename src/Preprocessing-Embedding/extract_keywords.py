# OM SRI GANESHAYA NAMAHA
import pandas as pd
import os
import json
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords_tfidf(texts, top_k=5):
    
    #Extract top-k TF-IDF keywords for each document.
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # allow multi-word phrases
        stop_words='english'
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    keywords_list = []

    for row in tfidf_matrix:
        row_data = row.toarray().flatten()
        top_indices = row_data.argsort()[-top_k:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        keywords_list.append(keywords)

    return keywords_list


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "../../data/embedding_data/embeddings.csv")
    output_path = os.path.join(base_dir, "../../data/cooking_data/cooking.csv")

    print(f"Loading embedded file to {input_path}")
    df = pd.read_csv(input_path)

    if "full_text" not in df.columns:
        raise ValueError("full_text column not found in embeddings.csv")

    print("Extracting TF-IDF keywords...")
    df["keywords"] = extract_keywords_tfidf(df["full_text"].fillna(""))

    # Convert list to JSON string for storage
    df["keywords"] = df["keywords"].apply(lambda x: json.dumps(x))

    print(f"Saving updated file to {output_path}")
    df.to_csv(output_path, index=False)

    print("DONE — keywords added successfully!")


if __name__ == "__main__":
    main()