import os, tarfile, urllib.request

def download_and_unpack(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    fname = url.split('/')[-1]
    path = os.path.join(target_dir, fname)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    with tarfile.open(path) as tar:
        tar.extractall(path=target_dir)
    print(f"Downloaded & unpacked {fname}")

if __name__ == "__main__":
    download_and_unpack(
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "../data/cifar-10"
    )
    download_and_unpack(
        "http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz",
        "../data/20news"
    )