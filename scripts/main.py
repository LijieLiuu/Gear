import os
import subprocess
import sys

STEPS = [
    ["python", "preprocess.py"],
    ["python", "split_dataset.py"],
    ["python", "train_baseline.py", "--epochs", "20"],
    ["python", "train_ssl.py", "--epochs", "100"],
    ["python", "train_ssl.py", "--epochs", "100", "--with_synth", "--synth_dir", "data/synth/50k"],
    ["python", "score_synthetics.py", "--input_dir", "data/synth/50k", "--output_dir", "data/synth/top10k",
     "--ckpt", "checkpoints/baseline_resnet18.pth", "--topk", "10000", "--min_conf", "0.90"],
    ["python", "train_ssl.py", "--epochs", "100", "--with_synth", "--synth_dir", "data/synth/top10k"],
    ["python", "plot_curves.py"],
]

def run_steps(steps):
    for cmd in steps:
        print("\n==> RUN:", " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            print("Step failed:", " ".join(cmd))
            sys.exit(ret)

if __name__ == "__main__":
    run_steps(STEPS)