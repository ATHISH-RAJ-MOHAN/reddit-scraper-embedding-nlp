# Reddit Scraper + NLP Pipeline

This project implements a complete end-to-end NLP pipeline that:

- Scrapes Reddit (HTML-based, no API key required)
- Cleans and preprocesses text
- Generates sentence embeddings
- Extracts TF-IDF keywords
- Clusters posts into meaningful topics
- Automates the entire workflow

All scraping is performed using the Old Reddit HTML interface.

---

## Project Overview

The system consists of the following stages:

### 1 Scraping (HTML-Based)
- Subreddits: Cooking, Recipes, AskCulinary, Baking, FoodScience
- Two-phase scraping:
  - Phase 1: Extract listing metadata (title, author, timestamp, permalink)
  - Phase 2: Visit each permalink to extract full post body and top comment
- Handles pagination, rate limiting, retries, and session persistence
- Saves raw data to:
  
  `data/parsed_json/`

---

### 2 Preprocessing
- Removes URLs, HTML tags, punctuation, numbers
- Converts text to lowercase
- Removes stopwords
- Lemmatizes words
- Combines title + body + comment into `full_text`
- Masks authors (e.g., author_1)
- Saves cleaned CSV files to:
  
  `data/cleaned_csv/`

---

### 3 Embedding Generation
- Model: all-MiniLM-L6-v2 (SentenceTransformers)
- Generates dense vector embeddings
- Stores embeddings as JSON lists inside CSV
- Output:
  
  `data/embedding_data/embeddings.csv`

---

### 4 TF-IDF Keyword Extraction
- Extracts top-k keywords per post (default: 5)
- Output:
  
  `data/embedding_data/embeddings_with_keywords.csv`

---

### 5 Clustering
- Algorithm: K-Means (default k=8)
- Identifies cluster keywords
- Finds representative posts
- Generates PCA visualization

Outputs:
- `data/clustered/clustered_messages.csv`
- `data/clustered/cluster_summary.json`
- `data/clustered/cluster_plot.png`

---

### 6 Automation
Runs scraping → preprocessing → embedding → clustering on a scheduled loop.

Script:
`src/automation/run_pipeline.py`

---

## Project Structure

```
project_root/
├── data/
│   ├── cleaned_csv/
│   ├── clustered/
│   ├── cooking_data/
│   ├── embedding_data/
│   └── parsed_json/
├── src/
│   ├── analysis/
│   ├── automation/
│   ├── config/
│   ├── Preprocessing-Embedding/
│   └── scraper/
├── README.md
└── requirements.txt
```

---

## How to Run the Full Pipeline

### 1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 2 Run the Scraper
```bash
python src/scraper/scrape_reddit.py \
  --subreddits Cooking,Baking,AskCulinary,FoodScience,Recipes \
  --limit 100
```
Raw JSON output: `data/parsed_json/`

### 3 Run Preprocessing
```bash
cd src/Preprocessing-Embedding
python preprocessing_pipeline.py
```
Output: `data/cleaned_csv/`

### 4 Generate Embeddings
```bash
python src/Preprocessing-Embedding/embed.py
```
Output: `data/embedding_data/embeddings.csv`

### 5 Extract TF-IDF Keywords
```bash
python src/Preprocessing-Embedding/extract_keywords.py
```
Output: `data/embedding_data/embeddings_with_keywords.csv`

### 6 Run Clustering
```bash
python src/analysis/cluster_messages.py --k 8
```
Outputs:
- `data/clustered/clustered_messages.csv`
- `data/clustered/cluster_summary.json`
- `data/clustered/cluster_plot.png`

### 7 Run the Automated Pipeline
```bash
python src/automation/run_pipeline.py 5 \
  --limit 100 \
  --subreddits Cooking,Baking,AskCulinary,FoodScience,Recipes \
  --k 8
```

---

## Final Combined Dataset
`data/cooking_data/cooking.csv`

---

## Team Members
- Neil Bai  
- Aadarsh Sudhir Ghiya  
- Athish Raj Mohan  

---

