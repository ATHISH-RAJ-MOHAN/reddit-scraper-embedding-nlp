import argparse
import os
import sys
import threading
import time
import traceback
from datetime import datetime

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from analysis.cluster_messages import run_clustering


def _script_path(*parts):
    return os.path.join(SRC_DIR, *parts)


def run_python_script(script_path, label):
    import subprocess

    print(f"[{label}] Running {script_path}")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    if result.stdout.strip():
        print(result.stdout)


def run_scraper(subreddits, limit):
    from scraper.scrape_reddit import scrape_all

    print("[Scrape] Fetching data...")
    scrape_all(subreddits=subreddits, limit=limit)


def run_pipeline_once(config):
    try:
        print("\n=== Pipeline Run Started ===")
        run_scraper(config.subreddits, config.limit)
        print("[Preprocess] Cleaning data...")
        run_python_script(_script_path("Preprocessing-Embedding", "preprocess.py"), "Preprocess")
        print("[Embed] Generating embeddings...")
        run_python_script(_script_path("Preprocessing-Embedding", "embed.py"), "Embedding")
        print("[Cluster] Clustering messages...")
        index, outputs = run_clustering(
            k=config.k,
            top_n=config.top_n,
            keywords_top_k=config.keywords_top_k,
            plot=True,
        )
        print("[Storage] Cluster outputs updated.")
        print("=== Pipeline Run Finished ===\n")
        return index, outputs
    except Exception as exc:
        print("[ERROR] Pipeline failed:", exc)
        traceback.print_exc()
        return None, None


class ClusterState:
    def __init__(self):
        self.lock = threading.Lock()
        self.index = None
        self.last_run = None
        self.updating = False

    def set_updating(self, value):
        with self.lock:
            self.updating = value

    def update_index(self, index):
        with self.lock:
            self.index = index
            self.last_run = datetime.now()

    def snapshot(self):
        with self.lock:
            return self.index, self.last_run, self.updating


def pipeline_worker(stop_event, state, config):
    interval_seconds = config.interval_minutes * 60
    while not stop_event.is_set():
        start = time.time()
        state.set_updating(True)
        index, _ = run_pipeline_once(config)
        if index is not None:
            state.update_index(index)
        state.set_updating(False)

        elapsed = time.time() - start
        sleep_for = max(0, interval_seconds - elapsed)
        if stop_event.wait(sleep_for):
            break


def load_query_model():
    from sentence_transformers import SentenceTransformer

    print("[Query] Loading sentence transformer model...")
    return SentenceTransformer("all-MiniLM-L6-v2")


def format_message(row):
    title = str(row.get("title_clean", "")).strip()
    snippet = str(row.get("full_text", "")).strip()
    snippet = snippet[:180] + ("..." if len(snippet) > 180 else "")
    return title or snippet or "(no text)"


def save_query_plot(output_dir, scores, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(
        output_dir,
        f"query_cluster_{int(time.time())}.png",
    )

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(scores)), scores)
    plt.title(title)
    plt.xlabel("Message Rank")
    plt.ylabel("Similarity")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def interactive_loop(state, config):
    model = load_query_model()
    output_dir = os.path.join(REPO_DIR, "data", "clustered")

    print("Type a query to find the closest cluster.")
    print("Use 'quit' to exit.")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Exiting.")
            return

        index, last_run, updating = state.snapshot()
        if index is None:
            print("No cluster data yet. Wait for the first pipeline run to finish.")
            continue
        if updating:
            print("Update in progress. Using the latest available clusters.")

        query_embedding = model.encode(query)
        cluster_id, score = index.find_closest_cluster(query_embedding)
        keywords = index.keywords_by_cluster.get(cluster_id, [])

        print(f"\nClosest cluster: {cluster_id} (score={score:.4f})")
        if keywords:
            print(f"Cluster keywords: {', '.join(keywords)}")

        top_matches = index.top_messages_for_query(
            query_embedding, cluster_id, top_n=config.top_n
        )
        if not top_matches:
            print("No messages found in this cluster.")
            continue

        scores = []
        for rank, (idx, sim) in enumerate(top_matches, start=1):
            row = index.df.iloc[idx]
            msg = format_message(row)
            scores.append(sim)
            print(f"{rank}. {msg} (sim={sim:.4f})")

        fig_path = save_query_plot(
            output_dir,
            scores=scores,
            title=f"Query Similarity for Cluster {cluster_id}",
        )
        print(f"Graph saved to {fig_path}\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run scraping -> preprocessing -> embedding -> clustering on a schedule.")
    parser.add_argument("interval_minutes", type=int, help="Minutes between pipeline runs.")
    parser.add_argument("--limit", type=int, default=100, help="Posts per subreddit.")
    parser.add_argument(
        "--subreddits",
        type=str,
        default="Cooking,Baking,AskCulinary,FoodScience,Recipes",
        help="Comma-separated subreddit list.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of clusters.")
    parser.add_argument("--top-n", type=int, default=5, help="Top messages to show per cluster.")
    parser.add_argument(
        "--keywords-top-k", type=int, default=6, help="Top keywords per cluster."
    )
    return parser.parse_args()


def main():
    os.chdir(REPO_DIR)
    args = _parse_args()
    args.subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]

    state = ClusterState()
    stop_event = threading.Event()

    worker = threading.Thread(
        target=pipeline_worker,
        args=(stop_event, state, args),
        daemon=True,
    )
    worker.start()

    interactive_loop(state, args)
    stop_event.set()
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
