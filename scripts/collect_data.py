#!/usr/bin/env python3
"""
Data Collection Script for Waste Segregation Assistant

Collects real waste images from webcam for training the model.
Images are saved to data/raw/<category>/ with timestamps.
"""

from datetime import datetime
from pathlib import Path
import time

import cv2

CATEGORIES = ["recyclable", "compost", "landfill"]
DATA_DIR = Path("data/raw")
IMG_SIZE = (224, 224)


def setup_directories():
    """Create category directories if they don't exist."""
    for cat in CATEGORIES:
        (DATA_DIR / cat).mkdir(parents=True, exist_ok=True)


def get_next_index(category):
    """Get the next image index for a category."""
    cat_dir = DATA_DIR / category
    existing = list(cat_dir.glob("*.jpg"))
    indices = []
    for f in existing:
        try:
            idx = int(f.stem.split("_")[-1])
            indices.append(idx)
        except (ValueError, IndexError):
            pass
    return max(indices) + 1 if indices else 0


def collect_images(category, num_images=30, delay=1.0):
    """Collect images for a specific category."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print(f"\n=== Collecting {num_images} images for '{category}' ===")
    print("Press SPACE to capture, 'q' to quit early")
    print(f"Images will be saved to {DATA_DIR / category}/")

    count = 0
    start_idx = get_next_index(category)

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Show preview with overlay
        display = frame.copy()
        cv2.putText(
            display,
            f"Category: {category}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display,
            f"Captured: {count}/{num_images}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display,
            "SPACE: capture  |  q: quit",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.imshow("Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # Space to capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            idx = start_idx + count
            filename = f"{category}_{timestamp}_{idx:04d}.jpg"
            filepath = DATA_DIR / category / filename

            # Resize and save
            resized = cv2.resize(frame, IMG_SIZE)
            cv2.imwrite(str(filepath), resized)
            print(f"  Saved: {filename}")
            count += 1
            time.sleep(delay)

        elif key == ord("q"):
            print(f"Stopped early. Captured {count} images.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect waste images for training")
    parser.add_argument(
        "--category", choices=CATEGORIES + ["all"], default="all", help="Category to collect"
    )
    parser.add_argument("--num", type=int, default=30, help="Number of images per category")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between captures (seconds)")
    args = parser.parse_args()

    setup_directories()

    categories = CATEGORIES if args.category == "all" else [args.category]

    print("=" * 50)
    print("WASTE SEGREGATION DATA COLLECTION")
    print("=" * 50)
    print(f"Categories: {', '.join(categories)}")
    print(f"Images per category: {args.num}")
    print(f"Save location: {DATA_DIR.absolute()}")
    print("=" * 50)

    total = 0
    for cat in categories:
        total += collect_images(cat, args.num, args.delay)

    print(f"\n=== Complete! Total images collected: {total} ===")
    print("\nNext steps:")
    print("  1. Review images in data/raw/<category>/")
    print("  2. Remove any poor quality images")
    print("  3. Run training: python scripts/train_model.py")


if __name__ == "__main__":
    main()
