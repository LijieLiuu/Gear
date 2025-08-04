import os
import json
import random
import shutil
import numpy as np
import torch

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def copy_tree(src_dir: str, dst_dir: str, file_list=None):
    ensure_dir(dst_dir)
    if file_list is None:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), src_dir)
                dst_path = os.path.join(dst_dir, rel)
                ensure_dir(os.path.dirname(dst_path))
                shutil.copy2(os.path.join(root, fn), dst_path)
    else:
        for rel in file_list:
            src_path = os.path.join(src_dir, rel)
            dst_path = os.path.join(dst_dir, rel)
            ensure_dir(os.path.dirname(dst_path))
            shutil.copy2(src_path, dst_path)

def linear_ramp(start, end, cur_step, ramp_steps):
    if ramp_steps <= 0:
        return end
    t = min(max(cur_step / ramp_steps, 0.0), 1.0)
    return start + (end - start) * t