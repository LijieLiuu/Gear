import subprocess
import sys

def run_step(script_name):
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"Error: {script_name} exited with code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    steps = [
        "preprocess.py",
        "split_dataset.py",
        "extract_features.py",
        "build_graph.py",
        "run_ssl_experiments.py"
    ]
    for script in steps:
        run_step(script)
    print("\nAll steps completed successfully.")