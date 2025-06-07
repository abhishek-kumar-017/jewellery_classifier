import os
import kagglehub
import shutil


def download_dataset():
    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("sapnilpatel/tanishq-jewellery-dataset")

    print("Downloaded to:", path)

    # Copy to data/raw directory
    raw_data_dir = os.path.join("data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    # Copy all files into data/raw
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(raw_data_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print(f"Dataset copied to: {raw_data_dir}")


if __name__ == "__main__":
    download_dataset()
