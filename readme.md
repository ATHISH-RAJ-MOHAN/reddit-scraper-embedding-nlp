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
