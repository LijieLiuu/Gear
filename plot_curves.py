import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from utils import ensure_dir

def plot_baseline(csv_path, out_png):
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    plt.figure()
    plt.plot(epochs, df["train_loss"], label="Train Loss")
    plt.plot(epochs, df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("ResNet-18 Loss")
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_loss.png"), dpi=200)

    plt.figure()
    plt.plot(epochs, df["train_acc"], label="Train Acc")
    plt.plot(epochs, df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("ResNet-18 Accuracy")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

def main(args):
    ensure_dir("figs")
    plot_baseline("results/baseline_curves.csv", "figs/graph.png")
    print("Saved figs/graph.png and figs/graph_loss.png")

def parse_args():
    return argparse.Namespace()

if __name__ == "__main__":
    main(parse_args())