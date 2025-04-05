import os
from pathlib import Path
import cv2
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO

# Load the model once globally
model = YOLO('best.pt')

# Dataset paths
splits = ['train', 'val', 'test']
base_path = Path('dataset')
img_root = base_path / 'images'
label_root = base_path / 'labels'

# Create label directories
for split in splits:
    (label_root / split).mkdir(parents=True, exist_ok=True)

def process_image(args):
    img_path, split = args
    label_dir = label_root / split
    label_file = label_dir / (img_path.stem + '.txt')

    if label_file.exists():
        return f"[{split}] Skipped: {img_path.name}"

    img = cv2.imread(str(img_path))
    if img is None:
        return f"[{split}] Skipped (unreadable): {img_path.name}"

    results = model(img, verbose=False)
    boxes = results[0].boxes
    h, w = img.shape[:2]

    with open(label_file, 'w') as f:
        for box in boxes:
            cls = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert to YOLO format
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    return f"[{split}] Labeled: {img_path.name}"

if __name__ == '__main__':
    tasks = []
    split_counts = {split: 0 for split in splits}

    for split in splits:
        img_dir = img_root / split
        for img_path in img_dir.glob('*.*'):
            tasks.append((img_path, split))

    print(f"Starting auto-labeling using {cpu_count()} CPU cores...")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_image, tasks)

    for result in results:
        if 'Labeled' in result:
            split = result.split(']')[0][1:]
            split_counts[split] += 1
        print(result)

    print("\n--- Final Labeling Summary ---")
    for split in splits:
        print(f"{split.capitalize():5}: {split_counts[split]} new images labeled")
    print(f"Total: {sum(split_counts.values())} newly labeled")
