import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n => Running {script_name} ")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("\n[ERROR OUTPUT]")
        print(result.stderr)

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    scripts = [
        os.path.join(base, "preprocess.py"),
        os.path.join(base, "embed.py"),
        os.path.join(base, "extract_keywords.py")
    ]

    for script in scripts:
        run_script(script)

    print("\n=== ALL STEPS COMPLETED SUCCESSFULLY ===")