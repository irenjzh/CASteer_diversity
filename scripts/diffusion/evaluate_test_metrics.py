import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.create_dataset import load_json, save_json
from core.eval.test_metrics import (
    DEFAULT_IMAGE_REWARD_MODEL,
    DEFAULT_MPS_MODEL,
    DEFAULT_PICKSCORE_MODEL,
    DEFAULT_PICKSCORE_PROCESSOR,
    SUPPORTED_TEST_METRICS,
    TestMetricsEvaluator,
    add_baseline_deltas,
    discover_experiment_dirs,
    flatten_test_metrics_for_summary,
    write_metrics_json,
    write_summary_csv,
)


def main(args: argparse.Namespace) -> None:
    discovered = discover_experiment_dirs(
        args.output_root,
        split=args.split,
        variants=args.variants,
    )
    if not discovered:
        raise ValueError(
            f"No experiment directories with config.json found in {args.output_root} "
            f"for split={args.split!r} and variants={args.variants!r}"
        )

    evaluator = TestMetricsEvaluator(
        metrics=args.metrics,
        device=args.device,
        clip_model=args.clip_model,
        clip_weight=args.clip_weight,
        clip_image_size=args.clip_image_size,
        batch_size=args.batch_size,
        pickscore_model=args.pickscore_model,
        pickscore_processor=args.pickscore_processor,
        image_reward_model=args.image_reward_model,
        mps_model=args.mps_model,
    )

    summary_rows = []
    for item in discovered:
        experiment_dir = item["experiment_dir"]
        config = item["config"]
        metrics_path = os.path.join(experiment_dir, "test_metrics.json")

        if os.path.exists(metrics_path) and not args.overwrite:
            print(f"Loading existing metrics: {metrics_path}")
            metrics_payload = load_json(metrics_path)
        else:
            print(f"Evaluating {experiment_dir}")
            metrics_payload = evaluator.evaluate_experiment_dir(
                experiment_dir,
                file_format=config.get("file_format"),
            )
            write_metrics_json(experiment_dir, metrics_payload)

        row = flatten_test_metrics_for_summary(
            metrics_payload,
            config=config,
            experiment_dir=experiment_dir,
        )
        summary_rows.append(row)

    summary_rows = add_baseline_deltas(summary_rows, metrics=args.metrics)

    summary_csv_path = os.path.join(args.output_root, "test_metrics_summary.csv")
    summary_json_path = os.path.join(args.output_root, "test_metrics_summary.json")
    write_summary_csv(summary_rows, summary_csv_path)
    save_json(summary_json_path, summary_rows)

    print("Saved summary:")
    print(f"  CSV:  {summary_csv_path}")
    print(f"  JSON: {summary_json_path}")
    print("Rows:")
    for row in summary_rows:
        metric_summary = ", ".join(
            f"{metric}={row[f'{metric}_mean']:.4f}"
            for metric in args.metrics
            if f"{metric}_mean" in row
        )
        print(
            f"  variant={row['variant']} strength={row.get('strength')} "
            f"dir={row['experiment_dir']} {metric_summary}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate test-set generations for baseline/random/steering variants."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory produced by run_manifest_generation.py",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name to evaluate (default: test)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "random_steering", "best_steering"],
        help="Variant names to include in the comparison",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_TEST_METRICS,
        default=list(SUPPORTED_TEST_METRICS),
        help="Metrics to compute",
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
        help="CLIP backbone used for clip_score and vendi embeddings",
    )
    parser.add_argument(
        "--clip_weight",
        type=float,
        default=2.5,
        help="CLIPScore scaling factor",
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
        help="Batch size for CLIP and PickScore backbones",
    )
    parser.add_argument(
        "--pickscore_model",
        type=str,
        default=DEFAULT_PICKSCORE_MODEL,
        help="PickScore model checkpoint",
    )
    parser.add_argument(
        "--pickscore_processor",
        type=str,
        default=DEFAULT_PICKSCORE_PROCESSOR,
        help="PickScore processor checkpoint",
    )
    parser.add_argument(
        "--image_reward_model",
        type=str,
        default=DEFAULT_IMAGE_REWARD_MODEL,
        help="ImageReward model checkpoint",
    )
    parser.add_argument(
        "--mps_model",
        type=str,
        default=DEFAULT_MPS_MODEL,
        help="MPS checkpoint loaded through imscore",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute metrics even if test_metrics.json already exists",
    )
    main(parser.parse_args())
