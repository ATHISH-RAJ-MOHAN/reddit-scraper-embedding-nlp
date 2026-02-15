import json
import re
import os
import pandas as pd
import string
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup
nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Test Mode (run only 10 rows)
TEST_MODE = False
TEST_LIMIT = 10

# Text Cleaning Function
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = text.encode("ascii", "ignore").decode()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)

    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in STOPWORDS and len(word) > 2
    ]

    return " ".join(tokens)

# Author Masking
def mask_authors(df):
    unique_authors = df["author_masked"].unique()
    mapping = {auth: f"author_{i+1}" for i, auth in enumerate(unique_authors)}
    df["author"] = df["author_masked"].map(mapping)
    return df

# Process a Single JSON File
def process_json(json_path, output_csv):
    print(f"\nProcessing: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if TEST_MODE:
        data = data[:TEST_LIMIT]

    rows = []

    for post in tqdm(data, desc="Cleaning posts"):
        title = clean_text(post.get("title", ""))
        body = clean_text(post.get("body", ""))
        comment_chunks = []
        if "comments" in post:
            comment_chunks.append(" ".join(post.get("comments", [])))
        if "top_comment" in post:
            comment_chunks.append(post.get("top_comment", ""))
        comments = clean_text(" ".join(comment_chunks)) if comment_chunks else ""

        full_text = f"{title} {body} {comments}".strip()

        permalink = post.get("permalink", "")
        post_id = permalink.split("/")[4] if permalink else ""

        timestamp = post.get("timestamp", "")
        date = timestamp.split("T")[0] if "T" in timestamp else ""
        time = timestamp.split("T")[1].split("+")[0] if "T" in timestamp else ""

        rows.append({
            "post_id": post_id,
            "subreddit": os.path.basename(json_path).replace(".json", ""),
            "title_clean": title,
            "body_clean": body,
            "comments": comments,
            "comments_clean": comments,
            "full_text": full_text,
            "date": date,
            "time": time,
            "author_masked": post.get("author", "unknown")
        })

    df = pd.DataFrame(rows)
    df = mask_authors(df)
    df = df.drop(columns=["author_masked"])
    df.to_csv(output_csv, index=False)

    print(f"Saved cleaned CSV to {output_csv}")

# Main
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parsed_dir = os.path.join(base_dir, "../../data/parsed_json")
    cleaned_dir = os.path.join(base_dir, "../../data/cleaned_csv")
    os.makedirs(cleaned_dir, exist_ok=True)

    json_files = [
        os.path.join(parsed_dir, f)
        for f in os.listdir(parsed_dir)
        if f.endswith(".json")
    ]

    if not json_files:
        print(f"No JSON files found in {parsed_dir}")
    else:
        for jf in sorted(json_files):
            output = os.path.join(
                cleaned_dir,
                os.path.basename(jf).replace(".json", "_cleaned.csv"),
            )
            process_json(jf, output)
