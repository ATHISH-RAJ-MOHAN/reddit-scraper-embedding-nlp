import subprocess
import os
import sys

def run_preprocessor():
    script_path = os.path.join(os.path.dirname(__file__), "./Preprocessing-Embedding/preprocess.py")
    print(f"Running: {script_path}")

    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

    print("\n--- PREPROCESSOR OUTPUT ---")
    print(result.stdout)

    if result.stderr:
        print("\n--- ERRORS ---")
        print(result.stderr)

if __name__ == "__main__":
    run_preprocessor()