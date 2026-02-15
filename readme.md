# Reddit Scraper + NLP Pipeline

This project scrapes Reddit posts, preprocesses text, generates embeddings, and extracts TF‑IDF keywords to build a structured dataset for downstream NLP tasks.

---
## Installations
```
pip install -r requirements.txt
```

## Preprocessing Pipeline

`preprocess.py` loads raw JSON files, cleans text, masks authors, and saves cleaned CSVs into `data/cleaned_csv/`.


```bash
cd src/Preprocessing-Embedding
python preprocessing_pipeline.py


## Run the full pipeline:
project_root/
│
├── data/
│   ├── parsed_json/
│   │   ├── AskCulinary.json
│   │   ├── Baking.json
│   │   ├── Cooking.json
│   │   ├── FoodScience.json
│   │   └── Recipes.json
│   │
│   ├── cleaned_csv/          ← output of preprocess.py
│   ├── embedding_data/       ← output of embed.py
│   └── cooking_data/         ← output of extract_keywords.py
│
└── src/
    └── Preprocessing-Embedding/
        ├── preprocess.py
        ├── embed.py
        ├── extract_keywords.py
        └── preprocessing_pipeline.py

## Output files:
data/cleaned_csv/
│
├── AskCulinary_cleaned.csv
├── Baking_cleaned.csv
├── Cooking_cleaned.csv
├── FoodScience_cleaned.csv
└── Recipes_cleaned.csv
```
## Final Dataset:
data/cooking_data/cooking.csv

---
## Added: Clustering + Automation

**Scraper updates**
- Added CLI arguments `--subreddits` and `--limit`.
- Captures `top_comment` from each post detail page when available.

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

**Notes**
- The first preprocessing run will download NLTK resources (`stopwords`, `wordnet`).
