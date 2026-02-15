import argparse
import ast
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_K = 8
DEFAULT_TOP_N = 5
DEFAULT_KEYWORDS_TOP_K = 6


def _default_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(base_dir, "../../data/embedding_data/embeddings.csv")
    output_dir = os.path.join(base_dir, "../../data/clustered")
    return embeddings_path, output_dir


def _parse_embedding_column(series):
    return np.vstack(series.apply(ast.literal_eval).values)


def load_embeddings(embeddings_path):
    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    df = pd.read_csv(embeddings_path)
    if "embedding" not in df.columns:
        raise ValueError("embedding column not found in embeddings file")
    embeddings = _parse_embedding_column(df["embedding"])
    return df, embeddings


def cluster_embeddings(embeddings, k, random_state=42):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def top_keywords_by_cluster(texts, labels, top_k=DEFAULT_KEYWORDS_TOP_K):
    keywords_by_cluster = {}
    unique_clusters = sorted(set(labels))
    for cluster_id in unique_clusters:
        cluster_texts = [t for t, l in zip(texts, labels) if l == cluster_id]
        if not cluster_texts:
            keywords_by_cluster[cluster_id] = []
            continue
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        tfidf = vectorizer.fit_transform(cluster_texts)
        mean_tfidf = tfidf.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-top_k:][::-1]
        features = vectorizer.get_feature_names_out()
        keywords = [features[i] for i in top_indices]
        keywords_by_cluster[cluster_id] = keywords
    return keywords_by_cluster


def representative_indices(embeddings, labels, centers, top_n=DEFAULT_TOP_N):
    reps = {}
    unique_clusters = sorted(set(labels))
    for cluster_id in unique_clusters:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            reps[cluster_id] = []
            continue
        sims = cosine_similarity(
            embeddings[idx], centers[cluster_id].reshape(1, -1)
        ).flatten()
        top_local = idx[np.argsort(sims)[-top_n:][::-1]]
        reps[cluster_id] = top_local.tolist()
    return reps


def build_cluster_summary(df, labels, embeddings, centers, keywords_by_cluster, top_n):
    reps = representative_indices(embeddings, labels, centers, top_n=top_n)
    summary = []
    for cluster_id in sorted(set(labels)):
        indices = reps.get(cluster_id, [])
        messages = []
        for idx in indices:
            row = df.iloc[idx]
            messages.append(
                {
                    "post_id": row.get("post_id", ""),
                    "title": row.get("title_clean", ""),
                    "snippet": str(row.get("full_text", ""))[:200],
                }
            )
        summary.append(
            {
                "cluster_id": int(cluster_id),
                "size": int((labels == cluster_id).sum()),
                "keywords": keywords_by_cluster.get(cluster_id, []),
                "representative_messages": messages,
            }
        )
    return summary


def save_cluster_plot(embeddings, labels, keywords_by_cluster, output_path):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=10, cmap="tab10")
    plt.title("K-Means Clusters (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    for cluster_id in sorted(set(labels)):
        cluster_points = coords[labels == cluster_id]
        if cluster_points.size == 0:
            continue
        centroid = cluster_points.mean(axis=0)
        keywords = keywords_by_cluster.get(cluster_id, [])
        label = f"C{cluster_id}: {', '.join(keywords[:2])}"
        plt.text(centroid[0], centroid[1], label, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@dataclass
class ClusterIndex:
    df: pd.DataFrame
    embeddings: np.ndarray
    labels: np.ndarray
    kmeans: KMeans
    keywords_by_cluster: dict

    @property
    def centers(self):
        return self.kmeans.cluster_centers_

    def find_closest_cluster(self, query_embedding):
        sims = cosine_similarity(self.centers, query_embedding.reshape(1, -1)).flatten()
        cluster_id = int(np.argmax(sims))
        return cluster_id, float(sims[cluster_id])

    def top_messages_for_cluster(self, cluster_id, top_n=DEFAULT_TOP_N):
        idx = np.where(self.labels == cluster_id)[0]
        if len(idx) == 0:
            return []
        sims = cosine_similarity(
            self.embeddings[idx], self.centers[cluster_id].reshape(1, -1)
        ).flatten()
        top_local = idx[np.argsort(sims)[-top_n:][::-1]]
        return top_local.tolist()

    def top_messages_for_query(self, query_embedding, cluster_id, top_n=DEFAULT_TOP_N):
        idx = np.where(self.labels == cluster_id)[0]
        if len(idx) == 0:
            return []
        sims = cosine_similarity(
            self.embeddings[idx], query_embedding.reshape(1, -1)
        ).flatten()
        top_local = idx[np.argsort(sims)[-top_n:][::-1]]
        return [(i, float(sims[j])) for j, i in enumerate(top_local)]


def run_clustering(
    embeddings_path=None,
    output_dir=None,
    k=DEFAULT_K,
    top_n=DEFAULT_TOP_N,
    keywords_top_k=DEFAULT_KEYWORDS_TOP_K,
    plot=True,
):
    if embeddings_path is None or output_dir is None:
        default_embeddings, default_output = _default_paths()
        embeddings_path = embeddings_path or default_embeddings
        output_dir = output_dir or default_output

    df, embeddings = load_embeddings(embeddings_path)
    labels, kmeans = cluster_embeddings(embeddings, k)
    keywords_by_cluster = top_keywords_by_cluster(
        df["full_text"].fillna("").tolist(), labels, top_k=keywords_top_k
    )

    df_out = df.copy()
    df_out["cluster_id"] = labels

    os.makedirs(output_dir, exist_ok=True)
    clustered_csv = os.path.join(output_dir, "clustered_messages.csv")
    df_out.to_csv(clustered_csv, index=False)

    summary = build_cluster_summary(
        df_out, labels, embeddings, kmeans.cluster_centers_, keywords_by_cluster, top_n
    )
    summary_path = os.path.join(output_dir, "cluster_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_path = None
    if plot:
        plot_path = os.path.join(output_dir, "cluster_plot.png")
        save_cluster_plot(embeddings, labels, keywords_by_cluster, plot_path)

    index = ClusterIndex(
        df=df_out,
        embeddings=embeddings,
        labels=labels,
        kmeans=kmeans,
        keywords_by_cluster=keywords_by_cluster,
    )
    return index, {"clustered_csv": clustered_csv, "summary_path": summary_path, "plot_path": plot_path}


def _parse_args():
    parser = argparse.ArgumentParser(description="Cluster message embeddings and summarize clusters.")
    default_embeddings, default_output = _default_paths()
    parser.add_argument("--embeddings", type=str, default=default_embeddings)
    parser.add_argument("--output-dir", type=str, default=default_output)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--keywords-top-k", type=int, default=DEFAULT_KEYWORDS_TOP_K)
    parser.add_argument("--no-plot", action="store_true", help="Skip cluster plot generation.")
    return parser.parse_args()


def main():
    args = _parse_args()
    plot = not args.no_plot
    index, outputs = run_clustering(
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        k=args.k,
        top_n=args.top_n,
        keywords_top_k=args.keywords_top_k,
        plot=plot,
    )
    print(f"Clusters saved to: {outputs['clustered_csv']}")
    print(f"Summary saved to: {outputs['summary_path']}")
    if outputs["plot_path"]:
        print(f"Plot saved to: {outputs['plot_path']}")
    print(f"Total clusters: {len(set(index.labels))}")


if __name__ == "__main__":
    main()
