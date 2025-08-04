import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from utils import ensure_dir

def main(seed=42, labeled_ratio=0.01):
    X = np.load("data/processed/cifar10/train_X.npy")
    y = np.load("data/processed/cifar10/train_y.npy")
    assert len(X) == 50000 and len(y) == 50000

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - labeled_ratio, random_state=seed)
    lab_idx, unlab_idx = next(sss.split(X, y))

    out_dir = ensure_dir("data/splits")
    np.save(os.path.join(out_dir, "X_lab.npy"), X[lab_idx])
    np.save(os.path.join(out_dir, "y_lab.npy"), y[lab_idx])
    np.save(os.path.join(out_dir, "X_unlab.npy"), X[unlab_idx])

    print(f"Labeled: {len(lab_idx)}; Unlabeled: {len(unlab_idx)}; saved to {out_dir}")

if __name__ == "__main__":
    main()