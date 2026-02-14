# OM SRI GANESHAYA NAMAHA
import subprocess
import os
import sys

def run_keyword_extractor():
    # Path to extract_keywords.py inside Preprocessing-Embedding
    script_path = os.path.join(os.path.dirname(__file__), "./Preprocessing-Embedding/extract_keywords.py")

    print(f"Running: {script_path}")

    # Use the venv python interpreter
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    print("\n--- KEYWORD EXTRACTION OUTPUT ---")
    print(result.stdout)

    if result.stderr:
        print("\n--- ERRORS ---")
        print(result.stderr)

if __name__ == "__main__":
    run_keyword_extractor()