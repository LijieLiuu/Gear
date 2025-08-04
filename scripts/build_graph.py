import pickle
import os
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def build_graph(feats, out_path, thresh=0.8):

    sim = cosine_similarity(feats)

    adj = (sim > thresh).astype(int)

    G = nx.from_numpy_array(adj)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {out_path}")

if __name__ == "__main__":
    feats = np.load("data/features/cifar10/features.npy")
    build_graph(
        feats,
        "data/graphs/cifar10_graph.gpkl",
        thresh=0.8
    )
    arr = np.load("data/processed/20news.npz", allow_pickle=True)
    docs = arr["docs"]
    vec = TfidfVectorizer(max_features=10000)
    text_feats = vec.fit_transform(docs).toarray()
    build_graph(
        text_feats,
        "data/graphs/20news_graph.gpkl",
        thresh=0.8
    )