"""
Compute CASteer steering vectors from paired positive/negative prompts.

For each prompt pair differing only by the target concept, we collect
cross-attention outputs, average across pairs per layer, and compute:

    steering_vector = normalize(mean(CA_pos) - mean(CA_neg))

Three prompt modes are supported:
  - concrete: "{ImageNet class} with {concept}" vs "{ImageNet class}"
  - style:    "{ImageNet class}, {concept} style" vs "{ImageNet class}"
  - human-related: "{person} {scene}, {concept}" vs "{person} {scene}"
"""
import argparse
import os
import pickle
from typing import Optional, Sequence, Tuple

import tqdm
from diffusers import DiffusionPipeline

from core.construct_prompts import get_prompts_concrete, get_prompts_human_related, get_prompts_style
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, run_image_model
from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode

from core.const import DEFAULT_BANK_CONCEPTS


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_pickle(payload, path: str) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'wb') as fout:
        pickle.dump(payload, fout, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, 'rb') as fin:
        return pickle.load(fin)


def collect_means(
    pipeline: DiffusionPipeline,
    model_name: str,
    prompts,
    device,
    progress_desc: str = "Collecting activations",
):
    """Run model on a list of prompts and return mean CA outputs per layer."""
    stats = CrossAttentionOutputStatsCollector(
        token_aggregation_mode=TokenAggregationMode.AVERAGE,
        normalize=False,
    )

    model_component = getattr(pipeline, 'transformer', None) or pipeline.unet
    hook_manager = diffusion_register_vector_controls_with_hooks(
        model_component,
        stats,
        model_type=DiffusionModelType.from_model(model_name),
    )

    try:
        for prompt in tqdm.tqdm(prompts, desc=progress_desc):
            _ = run_image_model(
                model_type=model_name,
                pipe=pipeline,
                prompt=prompt,
                seed=0,
                device=device,
            )
            stats.reset()
        return stats.means
    finally:
        hook_manager.remove_hooks()


def compute_steering_vectors(pos_means, neg_means):
    """
    Compute normalized steering vectors: normalize(pos_mean - neg_mean) per layer.
    """
    steering_vectors = {}

    for step in pos_means:
        steering_vectors[step] = {}
        for place in pos_means[step]:
            steering_vectors[step][place] = []
            for layer_idx in range(len(pos_means[step][place])):
                pos_vec = pos_means[step][place][layer_idx]
                neg_vec = neg_means[step][place][layer_idx]

                sv = pos_vec - neg_vec
                norm = sv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                sv = sv / norm

                steering_vectors[step][place].append(sv)

    return steering_vectors


def build_prompts_for_concept(
    concept: str,
    mode: str,
    num_prompts: int,
    concept_neg: Optional[str] = None,
):
    if mode == 'concrete':
        return get_prompts_concrete(
            num=num_prompts,
            concept_pos=concept,
            concept_neg=concept_neg,
        )
    if mode == 'style':
        return get_prompts_style(
            num=num_prompts,
            concept_pos=concept,
            concept_neg=concept_neg,
        )
    if mode == 'human-related':
        return get_prompts_human_related(
            concept_pos=concept,
            concept_neg=concept_neg,
        )
    raise ValueError(f"Unknown mode: {mode}")


def estimate_single_concept(
    pipeline: DiffusionPipeline,
    model_name: str,
    concept: str,
    mode: str,
    num_prompts: int,
    device,
    concept_neg: Optional[str] = None,
    progress_prefix: str = "",
):
    prompts_pos, prompts_neg = build_prompts_for_concept(
        concept=concept,
        mode=mode,
        num_prompts=num_prompts,
        concept_neg=concept_neg,
    )

    prefix = f"{progress_prefix} " if progress_prefix else ""
    print(f"{prefix}Generated {len(prompts_pos)} prompt pairs for concept '{concept}' (mode: {mode})")

    print(f"{prefix}Collecting positive prompt activations for '{concept}'...")
    pos_means = collect_means(
        pipeline=pipeline,
        model_name=model_name,
        prompts=prompts_pos,
        device=device,
        progress_desc=f"{concept} positive prompts",
    )

    print(f"{prefix}Collecting negative prompt activations for '{concept}'...")
    neg_means = collect_means(
        pipeline=pipeline,
        model_name=model_name,
        prompts=prompts_neg,
        device=device,
        progress_desc=f"{concept} negative prompts",
    )

    steering_vectors = compute_steering_vectors(pos_means, neg_means)
    print(f"{prefix}Computed steering vectors for concept '{concept}'")
    return steering_vectors


def get_default_bank_path(output_dir: str) -> str:
    return os.path.join(output_dir, 'default_bank.pickle')


def is_default_bank_complete(
    concept_bank,
    concepts: Sequence[Tuple[str, str]],
) -> bool:
    return all(concept in concept_bank for concept, _ in concepts)


def load_or_compute_default_bank(
    pipeline: DiffusionPipeline,
    model_name: str,
    output_dir: str,
    num_prompts: int,
    device,
    concepts: Optional[Sequence[Tuple[str, str]]] = None,
):
    concepts = list(concepts or DEFAULT_BANK_CONCEPTS)
    bank_path = get_default_bank_path(output_dir)
    concept_bank = {}

    if os.path.exists(bank_path):
        concept_bank = load_pickle(bank_path)
        print(
            f"Loaded existing default bank from {bank_path} "
            f"({len(concept_bank)}/{len(concepts)} concepts)"
        )

    total_concepts = len(concepts)
    for idx, (concept, mode) in enumerate(concepts, start=1):
        progress_prefix = f"[{idx}/{total_concepts}]"
        if concept in concept_bank:
            print(f"{progress_prefix} Concept '{concept}' (mode: {mode}) is already in the bank. Skipping.")
            continue

        print(f"{progress_prefix} Processing concept '{concept}' (mode: {mode})")
        concept_bank[concept] = estimate_single_concept(
            pipeline=pipeline,
            model_name=model_name,
            concept=concept,
            mode=mode,
            num_prompts=num_prompts,
            device=device,
            progress_prefix=progress_prefix,
        )
        save_pickle(concept_bank, bank_path)
        print(
            f"{progress_prefix} Saved bank progress to {bank_path} "
            f"({len(concept_bank)}/{total_concepts} concepts)"
        )

    return concept_bank, bank_path


def main(args):
    ensure_dir(args.output_dir)

    if args.run_mode == 'single':
        output_path = os.path.join(args.output_dir, f"{args.concept}.pickle")
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping.")
            return
    else:
        bank_path = get_default_bank_path(args.output_dir)
        if os.path.exists(bank_path):
            concept_bank = load_pickle(bank_path)
            if is_default_bank_complete(concept_bank, DEFAULT_BANK_CONCEPTS):
                print(
                    f"Default bank already exists and is complete: {bank_path} "
                    f"({len(concept_bank)}/{len(DEFAULT_BANK_CONCEPTS)} concepts)"
                )
                return
            print(
                f"Default bank exists but is incomplete: {bank_path} "
                f"({len(concept_bank)}/{len(DEFAULT_BANK_CONCEPTS)} concepts). Resuming computation."
            )

    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)
    device = get_device()

    if args.run_mode == 'default_bank':
        concept_bank, bank_path = load_or_compute_default_bank(
            pipeline=pipeline,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_prompts=args.num_prompts,
            device=device,
        )
        print(f"Saved default concept bank with {len(concept_bank)} concepts to {bank_path}")
        return

    steering_vectors = estimate_single_concept(
        pipeline=pipeline,
        model_name=args.model_name,
        concept=args.concept,
        mode=args.mode,
        num_prompts=args.num_prompts,
        device=device,
        concept_neg=args.concept_neg,
        progress_prefix="[1/1]",
    )
    save_pickle(steering_vectors, output_path)
    print(f"Saved steering vectors to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CASteer steering vectors from paired prompts")
    parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True,
                        help='Diffusion model to use for collecting activations')
    parser.add_argument('--run_mode', type=str, choices=['single', 'default_bank'], default='single',
                        help='Whether to compute a single concept or the default concept bank')
    parser.add_argument('--concept', type=str,
                        help='Target concept to compute steering vector for')
    parser.add_argument('--concept_neg', type=str, default=None,
                        help='Negative concept (default: prompt without any concept)')
    parser.add_argument('--mode', type=str, choices=['concrete', 'style', 'human-related'], default='concrete',
                        help='Prompt template mode')
    parser.add_argument('--num_prompts', type=int, default=50,
                        help='Number of prompt pairs (for concrete/style modes)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for steering vectors')

    args = parser.parse_args()
    if args.run_mode == 'single' and not args.concept:
        parser.error('--concept is required when --run_mode=single')
    main(args)
