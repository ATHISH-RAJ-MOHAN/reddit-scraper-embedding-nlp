# Reddit Scraper + NLP Pipeline

This project scrapes Reddit posts, preprocesses text, generates embeddings, and extracts TF‑IDF keywords to build a structured dataset for downstream NLP tasks.

---
## Installations
```
pip install -r requirements.txt
```

## Data Collection

The scraper collects posts from the Cooking, Recipes, AskCulinary, Baking, and FoodScience subreddits using the old Reddit HTML interface (no API key required). For each post, it retrieves the title, author, timestamp, permalink, self‑text, and the top comment. Pagination is handled automatically to fetch up to the requested number of posts (e.g., 1000). The raw output is saved as JSON files in data/parsed_json/.


## Preprocessing Pipeline

`preprocess.py` loads raw JSON files, cleans text, masks authors, and saves cleaned CSVs into `data/cleaned_csv/`.


```bash
cd src/Preprocessing-Embedding
python preprocessing_pipeline.py


## Run the full pipeline:
project_root/
├── data/
│   ├── cleaned_csv/
│   │   ├── AskCulinary_cleaned.csv
│   │   ├── Baking_cleaned.csv
│   │   ├── Cooking_cleaned.csv
│   │   ├── FoodScience_cleaned.csv
│   │   └── Recipes_cleaned.csv
│   ├── clustered/
│   │   ├── cluster_plot.png
│   │   ├── cluster_summary.json
│   │   └── clustered_messages.csv
│   ├── cooking_data/
│   │   └── cooking.csv
│   ├── embedding_data/
│   │   └── embeddings.csv
│   └── parsed_json/
│       ├── AskCulinary.json
│       ├── Baking.json
│       ├── Cooking.json
│       ├── FoodScience.json
│       └── Recipes.json
├── src/
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── cluster_messages.py
│   ├── automation/
│   │   ├── __init__.py
│   │   └── run_pipeline.py
│   ├── config/
│   │   └── headers.py
│   ├── Preprocessing-Embedding/
│   │   ├── embed.py
│   │   ├── extract_keywords.py
│   │   ├── preprocess.py
│   │   └── preprocessing_pipeline.py
│   └── scraper/
│       ├── fetch_html.py
│       ├── parse_posts.py
│       ├── scrape_reddit.py
│       └── test_scraper.py
├── readme.md
└── requirements.txt
```
## Final Dataset:
data/cooking_data/cooking.csv

---

Example:
```bash
python src/scraper/scrape_reddit.py --subreddits Cooking,Baking,AskCulinary,FoodScience,Recipes --limit 100
```

**Preprocessing updates**
- Merges `top_comment` and any `comments` list into a single cleaned field.
- Auto‑discovers all JSON files in `data/parsed_json/`.
- Writes `comments_clean` alongside `full_text`.

**Embedding updates**
- Auto‑discovers all cleaned CSVs in `data/cleaned_csv/`.
- Ensures embedding output directories exist.

**Clustering**
```bash
python src/analysis/cluster_messages.py --k 8
```
Outputs:
- `data/clustered/clustered_messages.csv`
- `data/clustered/cluster_summary.json`
- `data/clustered/cluster_plot.png`

**Automation**
```bash
python src/automation/run_pipeline.py 5 --limit 100 --subreddits Cooking,Baking,AskCulinary,FoodScience,Recipes --k 8
```
