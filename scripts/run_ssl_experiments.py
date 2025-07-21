import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RESULTS_PATH = "results/ssl_results.txt"

def write_result(line):
    with open(RESULTS_PATH, "a") as f:
        f.write(line + "\n")
    print(line)

def load_cifar(unlab_sample=2000):
    # 有标签特征（ResNet-18 512-dim）
    X_lab = np.load("data/features/cifar10/features.npy")
    y_lab = np.load("data/splits/cifar10_y_lab.npy")
    # 无标签特征
    X_unlab = np.load("data/features/cifar10/unlab_features.npy")
    # 抽样加速
    if unlab_sample and X_unlab.shape[0] > unlab_sample:
        idx = np.random.choice(X_unlab.shape[0], size=unlab_sample, replace=False)
        X_unlab = X_unlab[idx]
    # 图结构
    G = pickle.load(open("data/graphs/cifar10_graph.gpkl", "rb"))
    # 测试集特征
    X_test = np.load("data/features/cifar10/test_features.npy")
    y_test = np.load("data/processed/cifar10/test_y.npy")
    return X_lab, y_lab, X_unlab, G, X_test, y_test

def model_self_training(X_lab, y_lab, X_unlab=None, threshold=0.8, max_iter=1000):
    if X_unlab is not None:
        X_all = np.vstack([X_lab, X_unlab])
        y_unlab = -1 * np.ones(X_unlab.shape[0], dtype=int)
        y_all = np.concatenate([y_lab, y_unlab])
    else:
        X_all, y_all = X_lab, y_lab

    write_result(f"Self-training on {X_all.shape[0]} samples, dim={X_all.shape[1]}")
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, solver='saga', n_jobs=-1)
    )
    self_train = SelfTrainingClassifier(base, threshold=threshold, max_iter=10)
    self_train.fit(X_all, y_all)
    return self_train

def model_label_propagation(G, y_lab):
    n = G.number_of_nodes()
    labels = -np.ones(n, int)
    labeled_idx = np.random.choice(range(n), size=len(y_lab), replace=False)
    labels[labeled_idx] = y_lab
    X_dummy = np.zeros((n, 1))
    lp = LabelPropagation(kernel="knn", n_neighbors=10)
    lp.fit(X_dummy, labels)
    return lp

def generate_pseudo_samples(model, X_unlab):
    probs = model.predict_proba(X_unlab)
    labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    return labels, confidences

def filter_by_confidence(X_unlab, labels, confidences, threshold=0.9):
    mask = confidences >= threshold
    return X_unlab[mask], labels[mask]

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    open(RESULTS_PATH, "w").close()

    write_result("Loading CIFAR-10 features and graph…")
    X_lab, y_lab, X_unlab, G, X_test, y_test = load_cifar(unlab_sample=2000)

    write_result("Starting Model 1 (Self-Training)…")
    st_model = model_self_training(X_lab, y_lab, X_unlab, threshold=0.8, max_iter=2000)
    write_result("Evaluating Model 1…")
    acc1 = accuracy_score(y_test, st_model.predict(X_test))
    f11 = f1_score(y_test, st_model.predict(X_test), average="macro")
    write_result(f"Model 1 — Acc: {acc1:.4f}, F1: {f11:.4f}")

    write_result("Starting Model 2 (Label Propagation)…")
    lp_model = model_label_propagation(G, y_lab)

    write_result("Generating pseudo-labels for Model 2…")
    pseudo_labels, confidences = generate_pseudo_samples(st_model, X_unlab)

    write_result("Starting Model 2 (Self-Training with all pseudo)…")
    st_model2 = model_self_training(X_lab, y_lab, X_unlab, threshold=0.8, max_iter=2000)
    write_result("Evaluating Model 2…")
    acc2 = accuracy_score(y_test, st_model2.predict(X_test))
    f12 = f1_score(y_test, st_model2.predict(X_test), average="macro")
    write_result(f"Model 2 — Acc: {acc2:.4f}, F1: {f12:.4f}")

    write_result("Filtering pseudo-samples for Model 3…")
    X_filt, y_filt = filter_by_confidence(X_unlab, pseudo_labels, confidences, threshold=0.9)

    write_result("Starting Model 3 (Self-Training with filtered pseudo)…")
    st_model3 = model_self_training(X_lab, y_lab, X_filt, threshold=0.8, max_iter=2000)
    write_result("Evaluating Model 3…")
    acc3 = accuracy_score(y_test, st_model3.predict(X_test))
    f13 = f1_score(y_test, st_model3.predict(X_test), average="macro")
    write_result(f"Model 3 — Acc: {acc3:.4f}, F1: {f13:.4f}")

    write_result("All semi-supervised experiments completed.")