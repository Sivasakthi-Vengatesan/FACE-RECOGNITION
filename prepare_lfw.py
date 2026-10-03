import argparse
import shutil
from pathlib import Path
from collections import Counter

from sklearn.datasets import fetch_lfw_people
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Prepare an LFW subset for FaceID Pro.")
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--min-images", type=int, default=20,
                        help="Minimum images required for an identity.")
    parser.add_argument("--max-identities", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=30,
                        help="Maximum images retained per identity.")
    args = parser.parse_args()

    if args.min_images < 2 or args.max_identities < 2 or args.max_images < args.min_images:
        raise ValueError("Use min-images >= 2, max-identities >= 2 and max-images >= min-images.")

    print("Downloading/reading LFW through scikit-learn...")
    lfw = fetch_lfw_people(
        min_faces_per_person=args.min_images,
        color=True,
        resize=1.0,
        funneled=True,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Keep the most represented identities, then cap samples to make the demo reproducible.
    counts = Counter(lfw.target)
    selected = [idx for idx, _ in counts.most_common(args.max_identities)]
    selected_set = set(selected)

    # Remove an old generated dataset so stale identities cannot survive a rerun.
    for child in out.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    saved = Counter()
    for image, target in zip(lfw.images, lfw.target):
        if int(target) not in selected_set:
            continue
        if saved[int(target)] >= args.max_images:
            continue

        name = lfw.target_names[int(target)]
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("_")
        person_dir = out / safe_name
        person_dir.mkdir(parents=True, exist_ok=True)

        # fetch_lfw_people returns float RGB images in [0.0, 1.0].
        if image.max() <= 1.0:
            image_uint8 = (image * 255.0).clip(0, 255).astype("uint8")
        else:
            image_uint8 = image.clip(0, 255).astype("uint8")

        Image.fromarray(image_uint8, mode="RGB").save(
            person_dir / f"{saved[int(target)]+1:03d}.jpg", quality=95
        )
        saved[int(target)] += 1

    total = sum(saved.values())
    print(f"\nPrepared {total} images across {len(saved)} identities.")
    for target_idx in selected:
        name = lfw.target_names[target_idx]
        if saved[target_idx]:
            print(f"  {name}: {saved[target_idx]} images")

    if len(saved) < 2:
        raise RuntimeError("Fewer than two identities were prepared.")

    print(f"\nDataset ready at: {out.resolve()}")
    print("Next step: python train.py --data dataset")


if __name__ == "__main__":
    main()
