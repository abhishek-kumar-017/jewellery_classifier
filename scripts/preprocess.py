import os
import shutil
import random
import csv
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Configuration
IMAGE_SIZE = (224, 224)
RAW_DIR = "data/raw"
OUTPUT_DIR = "data"
LOG_CSV = "preprocessing_log.csv"
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def clear_and_create(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def resize_and_save(src_path, dest_path):
    img = Image.open(src_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img.save(dest_path)


def preprocess():
    print("📦 Starting preprocessing...")

    # Setup output directories
    split_dirs = ['train', 'val', 'test']
    for split in split_dirs:
        clear_and_create(os.path.join(OUTPUT_DIR, split))

    log_rows = [("Class", "Total", "Train", "Val", "Test", "Skipped")]

    class_names = [
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ]

    for class_name in class_names:
        class_path = os.path.join(RAW_DIR, class_name)
        images = [
            f for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]

        total = len(images)
        skipped = 0

        random.shuffle(images)

        train_split = int(total * TRAIN_RATIO)
        val_split = int(total * VAL_RATIO)

        train_imgs = images[:train_split]
        val_imgs = images[train_split:train_split + val_split]
        test_imgs = images[train_split + val_split:]

        for split, img_list in zip(split_dirs,
                                   [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_name in tqdm(img_list,
                                 desc=f"{split.capitalize()} - {class_name}",
                                 ncols=100):
                src = os.path.join(class_path, img_name)
                dest = os.path.join(split_class_dir, img_name)
                try:
                    resize_and_save(src, dest)
                except (UnidentifiedImageError, Exception):
                    skipped += 1

        log_rows.append((class_name, total, len(train_imgs), len(val_imgs),
                         len(test_imgs), skipped))

    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)

    print(f"\n✅ Preprocessing complete! Log saved to {LOG_CSV}")


if __name__ == "__main__":
    preprocess()
