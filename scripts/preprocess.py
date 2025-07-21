import os
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(doc):
    doc = re.sub(r'<[^>]+>', ' ', doc)
    doc = re.sub(r'[^a-zA-Z ]+', ' ', doc)
    tokens = [w for w in doc.lower().split() if w not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

def preprocess_cifar(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for batch in range(1, 6):
        path = os.path.join(src_dir, f"data_batch_{batch}")
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        X = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        y = np.array(data[b'labels'])
        np.save(os.path.join(out_dir, f"cifar10_batch{batch}_X.npy"), X)
        np.save(os.path.join(out_dir, f"cifar10_batch{batch}_y.npy"), y)
    print("CIFAR-10 preprocessing done.")

def preprocess_20news(src_dir, out_file):
    docs, labels = [], []
    for idx, cat in enumerate(os.listdir(src_dir)):
        cat_dir = os.path.join(src_dir, cat)
        for fn in os.listdir(cat_dir):
            with open(os.path.join(cat_dir, fn), errors="ignore") as f:
                txt = f.read()
            docs.append(clean_text(txt))
            labels.append(idx)
    np.savez_compressed(out_file, docs=docs, labels=labels)
    print("20 Newsgroups preprocessing done.")

def preprocess_cifar_test(src_file, out_X, out_y):
    os.makedirs(os.path.dirname(out_X), exist_ok=True)
    with open(src_file, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    X = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    y = np.array(data[b'labels'])
    np.save(out_X, X)
    np.save(out_y, y)
    print("CIFAR-10 test preprocessing done.")

if __name__ == "__main__":
    preprocess_cifar("cifar-10-batches-py", "data/processed/cifar10")
    preprocess_20news("20news-18828", "data/processed/20news.npz")
    # Preprocess CIFAR-10 test batch
    preprocess_cifar_test(
        "cifar-10-batches-py/test_batch",
        "data/processed/cifar10/test_X.npy",
        "data/processed/cifar10/test_y.npy"
    )