import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.create_dataset import create_coco_split_manifests, describe_split_example, validate_split_manifests


def main(args: argparse.Namespace) -> None:
    payload = create_coco_split_manifests(
        coco_dir=args.coco_dir,
        output_dir=args.output_dir,
        validation_size=args.validation_size,
        test_size=args.test_size,
        split_seed=args.split_seed,
        seeds=args.seeds,
        overwrite=args.overwrite,
    )

    stats = validate_split_manifests(
        payload["validation_manifest"],
        payload["test_manifest"],
        validation_size=args.validation_size,
        test_size=args.test_size,
        images_per_prompt=len(args.seeds),
    )

    print("Created/loaded COCO splits:")
    print(f"  validation_manifest_path: {payload['validation_manifest_path']}")
    print(f"  test_manifest_path:       {payload['test_manifest_path']}")
    print(f"  validation_reference_dir: {payload['validation_reference_dir']}")
    print(f"  test_reference_dir:       {payload['test_reference_dir']}")
    print(f"  stats:                    {stats}")
    print(f"  validation_example:       {describe_split_example(payload['validation_manifest'], limit=1)}")
    print(f"  test_example:             {describe_split_example(payload['test_manifest'], limit=1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create deterministic validation/test COCO manifests.")
    parser.add_argument("--coco_dir", type=str, required=True, help="Directory containing COCO val2017 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where split manifests are stored")
    parser.add_argument("--validation_size", type=int, default=100, help="Number of validation prompts")
    parser.add_argument("--test_size", type=int, default=2000, help="Number of test prompts")
    parser.add_argument("--split_seed", type=int, default=42, help="Shuffle seed used to build the splits")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Per-prompt generation seeds stored in the manifests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild manifests even if they already exist on disk",
    )
    main(parser.parse_args())
