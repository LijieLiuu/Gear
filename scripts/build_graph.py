import pickle
import os
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def build_graph(feats, out_path, thresh=0.8):
    # 计算相似度矩阵
    sim = cosine_similarity(feats)
    # 根据阈值生成邻接矩阵（1 表示连边，0 表示不连）
    adj = (sim > thresh).astype(int)
    # 构建图
    G = nx.from_numpy_array(adj)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 保存图结构
    with open(out_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {out_path}")

if __name__ == "__main__":
    # CIFAR-10 图
    feats = np.load("data/features/cifar10/features.npy")
    build_graph(
        feats,
        "data/graphs/cifar10_graph.gpkl",
        thresh=0.8
    )
    # 20 Newsgroups 图
    arr = np.load("data/processed/20news.npz", allow_pickle=True)
    docs = arr["docs"]
    vec = TfidfVectorizer(max_features=10000)
    text_feats = vec.fit_transform(docs).toarray()
    build_graph(
        text_feats,
        "data/graphs/20news_graph.gpkl",
        thresh=0.8
    )