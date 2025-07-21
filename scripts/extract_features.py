# extract_features.py

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_img_feats(X):
    """
    Simple flatten-based feature extraction:
    Convert each image of shape (H, W, C) into a 1D vector.
    """
    N = X.shape[0]
    return X.reshape(N, -1)

if __name__ == "__main__":
    # CIFAR-10 train set
    X_cifar_lab = np.load("data/splits/cifar10_X_lab.npy")
    feats_cifar = extract_img_feats(X_cifar_lab)
    os.makedirs("data/features/cifar10", exist_ok=True)
    np.save("data/features/cifar10/features.npy", feats_cifar)
    print("CIFAR-10 train features saved to data/features/cifar10/features.npy")

    # CIFAR-10 unlabeled set
    X_cifar_unlab = np.load("data/splits/cifar10_X_unlab.npy")
    feats_cifar_unlab = extract_img_feats(X_cifar_unlab)
    np.save("data/features/cifar10/unlab_features.npy", feats_cifar_unlab)
    print("CIFAR-10 unlabeled features saved to data/features/cifar10/unlab_features.npy")

    # CIFAR-10 test set
    X_test = np.load("data/processed/cifar10/test_X.npy")
    feats_test = extract_img_feats(X_test)
    np.save("data/features/cifar10/test_features.npy", feats_test)
    print("CIFAR-10 test features saved to data/features/cifar10/test_features.npy")

    # 20 Newsgroups train set
    X_20news_lab = np.load("data/splits/20news_X_lab.npy", allow_pickle=True)
    vec = TfidfVectorizer(max_features=10000)
    feats_20news = vec.fit_transform(X_20news_lab).toarray()
    os.makedirs("data/features/20news", exist_ok=True)
    np.save("data/features/20news/features.npy", feats_20news)
    print("20news features saved to data/features/20news/features.npy")