import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.create_dataset import load_json, save_json
from core.generate_dataset import build_standard_variants, generate_dataset_variants
from core.utils import SUPPORTED_DIFFUSION_MODELS


def infer_split(manifest_path: str, manifest) -> str:
    if manifest:
        split = manifest[0].get("split")
        if split:
            return str(split)

    filename = os.path.basename(manifest_path).lower()
    if "validation" in filename or "val" in filename:
        return "validation"
    if "test" in filename:
        return "test"
    return "dataset"


def normalize_variants(args: argparse.Namespace) -> list[dict]:
    if not args.variants:
        return [{"name": "baseline", "baseline": True}]

    requested = list(args.variants)
    steering_required = any(name in {"best_steering", "random_steering"} for name in requested)
    if steering_required and args.steering_source is None:
        raise ValueError("--steering_source is required for best_steering or random_steering variants")

    if requested == ["all"]:
        if args.steering_source is None:
            return [{"name": "baseline", "baseline": True}]
        return build_standard_variants(
            steering_source=args.steering_source,
            strength=args.steering_strength,
            random_seed=args.random_seed,
        )

    variants = []
    for name in requested:
        if name == "baseline":
            variants.append({"name": "baseline", "baseline": True})
        elif name == "best_steering":
            variants.append(
                {
                    "name": "best_steering",
                    "baseline": False,
                    "steering_source": args.steering_source,
                    "strength": args.steering_strength,
                    "use_all_diffusion_steps": args.use_all_diffusion_steps,
                    "renormalize_output": not args.disable_output_renorm, 
                    "concept_seed": args.concept_seed,
                }
            )
        elif name == "random_steering":
            variants.append(
                {
                    "name": "random_steering",
                    "baseline": False,
                    "steering_source": args.steering_source,
                    "strength": 1,
                    "randomize_source": True,
                    "random_seed": args.random_seed,
                    "use_all_diffusion_steps": args.use_all_diffusion_steps,
                    "renormalize_output": not args.disable_output_renorm,
                    "concept_seed": args.concept_seed,
                }
            )
        else:
            raise ValueError(f"Unknown variant: {name}")
    return variants


def main(args: argparse.Namespace):
    manifest = load_json(args.manifest_path)
    if not isinstance(manifest, list):
        raise TypeError(f"Manifest at {args.manifest_path} must be a JSON list")

    if args.max_prompts is not None:
        manifest = manifest[:args.max_prompts]

    if not manifest:
        raise ValueError("Manifest is empty after applying --max_prompts")

    split = args.split or infer_split(args.manifest_path, manifest)
    variants = normalize_variants(args)

    print(f"Loaded {len(manifest)} prompts from {args.manifest_path}")
    print(f"Split: {split}")
    print(f"Variants: {[variant['name'] for variant in variants]}")

    results = generate_dataset_variants(
        manifest=manifest,
        output_root=args.output_dir,
        split=split,
        variants=variants,
        model_name=args.model_name,
        file_format=args.file_format,
        concept_seed=args.concept_seed,
    )

    summary = {
        "manifest_path": args.manifest_path,
        "output_dir": args.output_dir,
        "split": split,
        "model_name": args.model_name,
        "file_format": args.file_format,
        "steering_source": args.steering_source,
        "steering_strength": args.steering_strength,
        "variants": [variant["name"] for variant in variants],
        "max_prompts": args.max_prompts,
        "results": results,
    }
    save_json(os.path.join(args.output_dir, "run_summary.json"), summary)

    print("Generation complete.")
    for row in results:
        print(
            f"  {row['variant']}: generated={row['generated_images']} "
            f"skipped={row['skipped_images']} dir={row['experiment_dir']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate validation/test images from a manifest with steering variants")

    parser.add_argument(
        "--model_name",
        type=str,
        choices=SUPPORTED_DIFFUSION_MODELS,
        required=True,
        help="Diffusion model name used for generation",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to validation_manifest.json or test_manifest.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where variant outputs should be written",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split name override (default: inferred from manifest contents or filename)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["all", "baseline", "best_steering", "random_steering"],
        default=["all"],
        help="Which variants to generate. Default: all (or baseline only if steering source is omitted).",
    )
    parser.add_argument(
        "--steering_source",
        type=str,
        default=None,
        help="Path to a steering vector pickle or a concept-bank pickle such as default_bank.pickle",
    )
    parser.add_argument(
        "--steering_strength",
        type=float,
        default=1.0,
        help="Additive steering strength used for best_steering and random_steering",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=12345,
        help="Seed used to derive the random steering vector",
    )
    parser.add_argument(
        "--concept_seed",
        type=int,
        default=0,
        help="Seed controlling deterministic concept selection from a concept bank",
    )
    parser.add_argument(
        "--use_all_diffusion_steps",
        action="store_true",
        help="Use stored per-step steering vectors instead of reusing step 0 for all diffusion steps",
    )
    parser.add_argument(
        "--disable_output_renorm",
        action="store_true",
        help="Disable output renormalization in the additive steering controller",
    )
    parser.add_argument(
        "--file_format",
        type=str,
        choices=["PNG", "JPEG"],
        default="PNG",
        help="File format for saved images",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Generate only the first N prompts from the manifest",
    )

    args = parser.parse_args()
    main(args)
