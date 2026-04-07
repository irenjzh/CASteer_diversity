"""
Evaluate CASteer on the I2P (Inappropriate Image Prompts) benchmark.

Downloads the I2P dataset (4,703 prompts) from HuggingFace and generates
images with optional steering applied. Results can then be evaluated with
NudeNet or Q16 classifiers.

Usage:
    # Generate without steering (baseline)
    python scripts/diffusion/run_i2p_eval.py \
        --model_name sd14 --output_dir ./results/i2p/baseline

    # Generate with nudity erasure
    python scripts/diffusion/run_i2p_eval.py \
        --model_name sd14 --output_dir ./results/i2p/casteer-2.0 \
        --steering_strength 2.0 --intermediate_clipping \
        --concept_path ./results/sd14/steering_vectors/nudity.pt
"""
import argparse
import os

import torch
from datasets import load_dataset

from core.controller import CrossAttentionOutputSteering
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.pickle import unpickle
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, run_image_model


def main(args):
    print("Loading I2P dataset...")
    ds = load_dataset("AIML-TUDA/i2p", split="train")
    print(f"Loaded {len(ds)} I2P prompts")

    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)
    device = get_device()

    vector_control = None
    if args.concept_path is not None:
        concept = unpickle(args.concept_path)
        vector_control = CrossAttentionOutputSteering(
            source_concepts=[concept],
            target_concepts=[None],
            strength=args.steering_strength,
            device=device,
            intermediate_clipping=args.intermediate_clipping,
            use_first_diffusion_step=True,
        )
        model_component = getattr(pipeline, 'transformer', None) or pipeline.unet
        diffusion_register_vector_controls_with_hooks(
            model_component,
            vector_control,
            model_type=DiffusionModelType.from_model(args.model_name),
        )

    os.makedirs(args.output_dir, exist_ok=True)
    skipped = generated = 0

    for i, row in enumerate(ds):
        output_path = os.path.join(args.output_dir, f"{i:05d}.png")
        if os.path.exists(output_path):
            skipped += 1
            continue

        images = run_image_model(
            model_type=args.model_name,
            pipe=pipeline,
            prompt=row['prompt'],
            seed=row['sd_seed'],
            device=device,
        )
        if vector_control is not None:
            vector_control.reset()

        images[0].save(output_path)
        generated += 1

        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(ds)}")

    print(f"Done. Generated {generated}, skipped {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run I2P evaluation with optional CASteer")
    parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--concept_path', type=str, default=None, help='Path to steering vector for erasure')
    parser.add_argument('--steering_strength', type=float, default=2.0)
    parser.add_argument('--intermediate_clipping', action='store_true')

    args = parser.parse_args()
    main(args)
