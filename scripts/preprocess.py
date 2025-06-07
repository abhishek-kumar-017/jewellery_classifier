import os
import shutil
import random
import csv
import yaml
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from tqdm import tqdm


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clear_and_create(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def apply_augmentation(img, config):
    # Horizontal Flip
    if config.get("horizontal_flip", False) and random.random() > 0.5:
        img = ImageOps.mirror(img)

    # Rotation
    degrees = config.get("rotation", 0)
    if degrees > 0:
        angle = random.uniform(-degrees, degrees)
        img = img.rotate(angle)

    # Color Jitter
    if config.get("color_jitter", False):
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    return img


def resize_and_save(src_path, dest_path, size, augment_config=None):
    img = Image.open(src_path).convert('RGB')
    if augment_config:
        img = apply_augmentation(img, augment_config)
    img = img.resize(size)
    img.save(dest_path)


def preprocess():
    config = load_config()

    IMAGE_SIZE = tuple(config["image_size"])
    RAW_DIR = config["data_paths"]["raw"]
    OUTPUT_DIR = config["data_paths"]["output"]
    VALID_EXTENSIONS = [ext.lower() for ext in config["valid_extensions"]]
    SPLIT = config["split_ratio"]

    print("📦 Starting preprocessing...")

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
        train_split = int(total * SPLIT["train"])
        val_split = int(total * SPLIT["val"])

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
                    aug = config["augmentation"][
                        "train"] if split == "train" else None
                    resize_and_save(src, dest, IMAGE_SIZE, augment_config=aug)
                except (UnidentifiedImageError, Exception) as e:
                    skipped += 1
                    print(f"Skipped {src}: {e}")

        log_rows.append((class_name, total, len(train_imgs), len(val_imgs),
                         len(test_imgs), skipped))

    # Write CSV log
    log_path = os.path.join(OUTPUT_DIR, "preprocessing_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)

    print(f"\n✅ Preprocessing complete! Log saved to {log_path}")


if __name__ == "__main__":
    preprocess()
