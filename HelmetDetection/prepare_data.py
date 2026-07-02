import random
import shutil
from pathlib import Path
from voc_to_yolo import parse_xml

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "data" / "images"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
OUT_DIR = BASE_DIR / "yolo_dataset"

CLASS_TO_ID = {
    "helmet": 0,
    "head": 1,
    "person": 2,
}

def process_split(xml_files, split_name):
    images_out_dir = OUT_DIR / "images" / split_name
    labels_out_dir = OUT_DIR / "labels" / split_name

    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    for xml_file in xml_files:
        filename, labels = parse_xml(xml_file, CLASS_TO_ID)
        src_img = IMAGES_DIR / filename
        dst_img = images_out_dir / filename
        shutil.copy2(src_img, dst_img)

        label_path = labels_out_dir / f"{Path(filename).stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            for class_id, x_center, y_center, box_w, box_h in labels:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

def write_data_yaml():
    names = [None]*len(CLASS_TO_ID)
    for name, idx in CLASS_TO_ID.items():
        names[idx] = name

    yaml_text = (
        f"path: {OUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        f"nc: {len(CLASS_TO_ID)}\n"
        f"names: {names}\n"
    )

    with open(OUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)

xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))

random.seed(42)
random.shuffle(xml_files)

split_idx = int(0.8*len(xml_files))
train_files = xml_files[:split_idx]
val_files = xml_files[split_idx:]

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

process_split(train_files, "train")
process_split(val_files, "val")
write_data_yaml()