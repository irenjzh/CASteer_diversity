"""Utilities for building deterministic validation/test splits from COCO val2017.

The split logic:
1. Load `captions_val2017.json` and `val2017/`.
2. Keep exactly one caption per unique COCO image.
3. Shuffle records with a fixed seed.
4. Take the first `validation_size` records for validation and the next
   `test_size` records for test.
5. Save both manifests and create reference image directories.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_SEEDS = [0, 1, 2, 3, 4]

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _download_if_missing(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    ensure_dir(os.path.dirname(path) or ".")
    urllib.request.urlretrieve(url, path)


def ensure_coco_val2017(coco_dir: str) -> Dict[str, str]:
    """Ensure MS-COCO val2017 images and captions are present on disk."""
    coco_dir = ensure_dir(coco_dir)
    annotations_path = os.path.join(coco_dir, "annotations", "captions_val2017.json")
    val_dir = os.path.join(coco_dir, "val2017")

    if not os.path.exists(annotations_path):
        annotations_zip = os.path.join(coco_dir, "annotations_trainval2017.zip")
        _download_if_missing(COCO_ANNOTATIONS_URL, annotations_zip)
        with zipfile.ZipFile(annotations_zip) as zf:
            zf.extractall(coco_dir)

    if not os.path.exists(val_dir):
        val_zip = os.path.join(coco_dir, "val2017.zip")
        _download_if_missing(COCO_VAL2017_URL, val_zip)
        with zipfile.ZipFile(val_zip) as zf:
            zf.extractall(coco_dir)

    return {
        "annotations_path": annotations_path,
        "val_dir": val_dir,
    }


def load_coco_caption_records(coco_dir: str) -> List[Dict[str, Any]]:
    """Load one caption per unique image from COCO val2017.

    COCO usually contains multiple captions per image. This function keeps the
    first caption encountered for each unique `image_id`, reproducing the
    original split-generation logic.
    """
    coco_paths = ensure_coco_val2017(coco_dir)
    coco_data = load_json(coco_paths["annotations_path"])
    image_to_caption: Dict[int, str] = {}

    for ann in coco_data["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in image_to_caption:
            image_to_caption[image_id] = ann["caption"]

    records: List[Dict[str, Any]] = []
    for image_id, caption in image_to_caption.items():
        records.append(
            {
                "image_id": image_id,
                "caption": caption,
                "real_image_path": os.path.join(coco_paths["val_dir"], f"{image_id:012d}.jpg"),
            }
        )
    return records


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        return
    try:
        shutil.copy2(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def create_reference_dir(records: Sequence[Dict[str, Any]], reference_dir: str) -> str:
    """Create a directory of real COCO images for FID/reference metrics."""
    ensure_dir(reference_dir)
    for record in records:
        src = record["real_image_path"]
        dst = os.path.join(reference_dir, os.path.basename(src))
        _link_or_copy(src, dst)
    return reference_dir


def create_coco_split_manifests(
    coco_dir: str,
    output_dir: str,
    validation_size: int = 100,
    test_size: int = 2000,
    split_seed: int = 42,
    seeds: Optional[Sequence[int]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build deterministic validation/test manifests from COCO val2017.

    Returns a payload with saved manifest paths, reference directories and
    in-memory manifest contents.
    """
    seeds = list(seeds or DEFAULT_SEEDS)
    split_dir = ensure_dir(os.path.join(output_dir, "splits"))
    validation_manifest_path = os.path.join(split_dir, "validation_manifest.json")
    test_manifest_path = os.path.join(split_dir, "test_manifest.json")

    if not overwrite and os.path.exists(validation_manifest_path) and os.path.exists(test_manifest_path):
        validation_manifest = load_json(validation_manifest_path)
        test_manifest = load_json(test_manifest_path)
    else:
        records = load_coco_caption_records(coco_dir)
        rng = random.Random(split_seed)
        rng.shuffle(records)

        required = validation_size + test_size
        if len(records) < required:
            raise ValueError(f"Need at least {required} COCO records, found {len(records)}")

        selected = records[:required]
        validation_records = selected[:validation_size]
        test_records = selected[validation_size:required]

        validation_manifest = [
            {
                "split": "validation",
                "prompt_index": idx,
                "image_id": record["image_id"],
                "caption": record["caption"],
                "real_image_path": record["real_image_path"],
                "seeds": seeds,
            }
            for idx, record in enumerate(validation_records)
        ]
        test_manifest = [
            {
                "split": "test",
                "prompt_index": idx,
                "image_id": record["image_id"],
                "caption": record["caption"],
                "real_image_path": record["real_image_path"],
                "seeds": seeds,
            }
            for idx, record in enumerate(test_records)
        ]

        save_json(validation_manifest_path, validation_manifest)
        save_json(test_manifest_path, test_manifest)

    validation_reference_dir = create_reference_dir(
        validation_manifest,
        os.path.join(split_dir, "validation_reference"),
    )
    test_reference_dir = create_reference_dir(
        test_manifest,
        os.path.join(split_dir, "test_reference"),
    )

    return {
        "validation_manifest_path": validation_manifest_path,
        "test_manifest_path": test_manifest_path,
        "validation_reference_dir": validation_reference_dir,
        "test_reference_dir": test_reference_dir,
        "validation_manifest": validation_manifest,
        "test_manifest": test_manifest,
    }


def validate_split_manifests(
    validation_manifest: Sequence[Dict[str, Any]],
    test_manifest: Sequence[Dict[str, Any]],
    validation_size: int = 100,
    test_size: int = 2000,
    images_per_prompt: int = 5,
) -> Dict[str, int]:
    """Validate split sizes, uniqueness and number of seeds per prompt."""
    val_ids = [record["image_id"] for record in validation_manifest]
    test_ids = [record["image_id"] for record in test_manifest]

    if len(set(val_ids)) != validation_size:
        raise ValueError("Validation split does not contain the expected number of unique image_id values")
    if len(set(test_ids)) != test_size:
        raise ValueError("Test split does not contain the expected number of unique image_id values")
    if set(val_ids) & set(test_ids):
        raise ValueError("Validation and test splits overlap")

    for manifest in (validation_manifest, test_manifest):
        for record in manifest:
            if len(record["seeds"]) != images_per_prompt:
                raise ValueError(
                    f'Expected {images_per_prompt} seeds per prompt, got {len(record["seeds"])} '
                    f'for {record["image_id"]}'
                )

    return {
        "validation_unique_image_ids": len(set(val_ids)),
        "test_unique_image_ids": len(set(test_ids)),
        "images_per_prompt": images_per_prompt,
    }


def describe_split_example(manifest: Sequence[Dict[str, Any]], limit: int = 2) -> List[Dict[str, Any]]:
    """Return a short preview of manifest entries for inspection/debugging."""
    return [dict(item) for item in list(manifest)[:limit]]


def count_coco_val_images(coco_dir: str) -> int:
    """Return the number of JPG files available in COCO val2017."""
    coco_paths = ensure_coco_val2017(coco_dir)
    return len(list(Path(coco_paths["val_dir"]).glob("*.jpg")))
