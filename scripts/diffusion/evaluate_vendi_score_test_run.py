import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.eval.test_metrics import DEFAULT_VARIANT_ORDER, evaluate_vendi_score_test_run


def main(args: argparse.Namespace) -> None:
    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(args.output_root, "vendi_score_test_run.json")

    payload = evaluate_vendi_score_test_run(
        args.output_root,
        output_path=output_path,
        split=args.split,
        variants=args.variants,
        device=args.device,
        clip_model=args.clip_model,
        clip_image_size=args.clip_image_size,
        batch_size=args.batch_size,
        q=args.q,
    )

    print(f"Saved Vendi Score JSON: {output_path}")
    for row in payload["summary"]:
        print(
            f"  {row['label']}: vendi_score={row['vendi_score_mean']:.4f} "
            f"std={row['vendi_score_std']:.4f} dir={row['experiment_dir']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Vendi Score for baseline, random_steering, and best_steering test outputs."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root test_run directory containing test variant experiment folders",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional JSON output path (default: <output_root>/vendi_score_test_run.json)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name to evaluate when config.json files are present",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANT_ORDER),
        help="Variant names to include (default: baseline random_steering best_steering)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Metric device override (for example: cuda, cpu, mps)",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        help="CLIP image backbone used to build the Vendi similarity matrix",
    )
    parser.add_argument(
        "--clip_image_size",
        type=int,
        default=224,
        help="CLIP image resolution",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for CLIP image embeddings",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=1.0,
        help="Vendi Score order q passed to vendi_score.vendi.score_K",
    )
    main(parser.parse_args())
