"""Generation helpers for validation/test manifests with steering variants.
"""

import csv
import glob
import os
import pickle
import re
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import torch

from .create_dataset import ensure_dir, save_json
from .diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from .diversity_controller import CrossAttentionOutputAdditiveSteering
from .pickle import unpickle
from .utils import get_device, init_pipeline_for_image_model, run_image_model

SAVE_OPTIONS = {
    "PNG": {},
    "JPEG": {
        "subsampling": "4:4:4",
        "quality": 95,
    },
}

EXTENSIONS = {
    "PNG": "png",
    "JPEG": "jpg",
}

DEFAULT_RANDOM_STEERING_SEED = 12345


def save_pickle(payload, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as fout:
        pickle.dump(payload, fout, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(path: str):
    import json

    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def choose_concept_name(concept_names: tp.Sequence[str], prompt_index: int, seed: int, concept_seed: int = 0) -> str:
    import random

    rng = random.Random(concept_seed + prompt_index * 1000 + seed)
    return concept_names[rng.randrange(len(concept_names))]


def assign_concepts_to_seeds(
    concept_names: tp.Sequence[str],
    prompt_index: int,
    seeds: tp.Sequence[int],
    concept_seed: int = 0,
) -> dict[str, str]:
    import random

    if not concept_names:
        return {}

    ordered_seeds = sorted(int(seed) for seed in seeds)
    assignments: dict[str, str] = {}
    shuffled_concepts = list(concept_names)
    rng = random.Random(concept_seed + prompt_index * 1000)
    rng.shuffle(shuffled_concepts)
    concept_count = len(shuffled_concepts)

    # Reuse a single deterministic permutation per prompt so concepts do not
    # repeat until the available list is exhausted.
    for index, seed in enumerate(ordered_seeds):
        assignments[str(seed)] = shuffled_concepts[index % concept_count]

    return assignments


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "item"


def build_variant_dir_name(split: str, variant: str, strength: float | None = None) -> str:
    parts = [split, variant]
    if strength is not None:
        parts.append(f"b{strength}")
    return _safe_name("_".join(parts))


def _prompt_dir(experiment_dir: str, prompt_index: int) -> str:
    return os.path.join(experiment_dir, f"prompt_{prompt_index:04d}")


def _prompt_image_path(experiment_dir: str, prompt_index: int, seed: int, file_format: str) -> str:
    ext = EXTENSIONS[file_format]
    return os.path.join(_prompt_dir(experiment_dir, prompt_index), f"seed{seed:02d}.{ext}")


def _is_prompt_complete(experiment_dir: str, prompt_index: int, seeds: tp.Sequence[int], file_format: str) -> bool:
    return all(os.path.exists(_prompt_image_path(experiment_dir, prompt_index, seed, file_format)) for seed in seeds)


def _save_prompt_metadata(
    prompt_dir: str,
    record: dict[str, tp.Any],
    concept_names: tp.Sequence[str],
    chosen_concepts: dict[str, str | None],
) -> None:
    save_json(
        os.path.join(prompt_dir, "metadata.json"),
        {
            "prompt_index": record["prompt_index"],
            "image_id": record.get("image_id"),
            "caption": record["caption"],
            "real_image_path": record.get("real_image_path"),
            "seeds": list(record["seeds"]),
            "concept_names": list(concept_names),
        },
    )
    save_json(os.path.join(prompt_dir, "chosen_concepts.json"), chosen_concepts)


def _is_concept_bank(payload: tp.Any) -> bool:
    return isinstance(payload, dict) and bool(payload) and isinstance(next(iter(payload.keys())), str)


def _is_steering_vectors(payload: tp.Any) -> bool:
    return isinstance(payload, dict) and bool(payload) and isinstance(next(iter(payload.keys())), int)


def load_steering_source(source: str | dict | None):
    if source is None:
        return None
    if isinstance(source, str):
        source = unpickle(source)
    if _is_concept_bank(source) or _is_steering_vectors(source):
        return source
    raise TypeError("Unsupported steering source format. Expected a concept bank or steering vector mapping.")


def template_from_source(source: dict) -> dict[int, dict[str, list[torch.Tensor]]]:
    if _is_steering_vectors(source):
        return source
    if _is_concept_bank(source):
        first_concept = sorted(source.keys())[0]
        return source[first_concept]
    raise TypeError("Cannot extract template vectors from unsupported steering source.")


def create_random_steering_vectors(
    template_vectors: dict[int, dict[str, list[torch.Tensor]]],
    seed: int = DEFAULT_RANDOM_STEERING_SEED,
) -> dict[int, dict[str, list[torch.Tensor]]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_vectors: dict[int, dict[str, list[torch.Tensor]]] = {}

    for step, place_payload in template_vectors.items():
        random_vectors[step] = {}
        for place_in_unet, layer_vectors in place_payload.items():
            random_vectors[step][place_in_unet] = []
            for layer_vector in layer_vectors:
                base = layer_vector.detach().cpu()
                noise = torch.randn(base.shape, generator=generator, dtype=torch.float32)
                norm = torch.linalg.norm(noise, dim=-1, keepdim=True).clamp(min=1e-8)
                noise = (noise / norm).to(base.dtype)
                random_vectors[step][place_in_unet].append(noise)
    return random_vectors


def build_standard_variants(
    steering_source: str | dict,
    strength: float,
    random_seed: int = DEFAULT_RANDOM_STEERING_SEED,
) -> list[dict[str, tp.Any]]:
    return [
        {
            "name": "best_steering",
            "baseline": False,
            "steering_source": steering_source,
            "strength": strength,
        },
        {
            "name": "baseline",
            "baseline": True,
            "strength": None,
        },
        {
            "name": "random_steering",
            "baseline": False,
            "steering_source": steering_source,
            "strength": strength,
            "randomize_source": True,
            "random_seed": random_seed,
        },
    ]


def _hook_fixed_steering(
    pipeline,
    model_name: str,
    steering_vectors: dict[int, dict[str, list[torch.Tensor]]],
    strength: float,
    device,
    use_all_diffusion_steps: bool,
    renormalize_output: bool,
):
    vector_control = CrossAttentionOutputAdditiveSteering(
        source_concepts=[steering_vectors],
        strength=strength,
        device=device,
        use_first_diffusion_step=not use_all_diffusion_steps,
        renormalize_output=renormalize_output,
        output_dtype=None,
    )
    model_component = getattr(pipeline, "transformer", None) or pipeline.unet
    hook_manager = diffusion_register_vector_controls_with_hooks(
        model_component,
        vector_control,
        model_type=DiffusionModelType.from_model(model_name),
    )
    return hook_manager, vector_control


def _generate_single_image(
    *,
    pipeline,
    model_name: str,
    prompt: str,
    seed: int,
    device,
    output_path: str,
    file_format: str,
    steering_vectors: dict[int, dict[str, list[torch.Tensor]]] | None = None,
    strength: float | None = None,
    use_all_diffusion_steps: bool = False,
    renormalize_output: bool = True,
    reuse_hook: tuple[tp.Any, tp.Any] | None = None,
) -> None:
    transient_hook = None
    transient_control = None

    try:
        if steering_vectors is not None and reuse_hook is None:
            transient_hook, transient_control = _hook_fixed_steering(
                pipeline=pipeline,
                model_name=model_name,
                steering_vectors=steering_vectors,
                strength=float(strength),
                device=device,
                use_all_diffusion_steps=use_all_diffusion_steps,
                renormalize_output=renormalize_output,
            )

        images = run_image_model(
            model_type=model_name,
            pipe=pipeline,
            prompt=prompt,
            seed=seed,
            device=device,
        )

        active_control = transient_control
        if reuse_hook is not None:
            _, active_control = reuse_hook
        if active_control is not None:
            active_control.reset()

        images[0].save(output_path, format=file_format, **SAVE_OPTIONS[file_format])
    finally:
        if transient_hook is not None:
            transient_hook.remove_hooks()


def _generated_file_pattern(file_format: str) -> str:
    return f"*.{EXTENSIONS[file_format]}"


def _iter_prompt_dirs(experiment_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(experiment_dir, "prompt_*")))


def collect_prompt_image_paths(prompt_dir: str, file_format: str = "PNG") -> list[str]:
    return sorted(glob.glob(os.path.join(prompt_dir, _generated_file_pattern(file_format))))


def evaluate_validation_experiment_dir(
    experiment_dir: str,
    reference_dir: str,
    file_format: str = "PNG",
    clip_model: str = "ViT-B/32",
    clip_weight: float = 2.5,
    clip_image_size: int = 224,
    clip_batch_size: int = 50,
    clip_device: str | None = None,
) -> dict[str, tp.Any]:
    from .eval.clip import clip_score
    from .eval.fid import compute_fid

    prompt_dirs = _iter_prompt_dirs(experiment_dir)
    if not prompt_dirs:
        raise ValueError(f"No prompt_* directories found in {experiment_dir}")

    image_paths: list[str] = []
    texts: list[str] = []
    prompt_slices: list[tuple[str, str, int, int]] = []
    cursor = 0

    for prompt_dir in prompt_dirs:
        prompt_name = os.path.basename(prompt_dir)
        prompt_path = os.path.join(prompt_dir, "prompt.txt")
        if not os.path.exists(prompt_path):
            continue
        prompt = open(prompt_path, "r", encoding="utf-8").read().strip()
        prompt_images = collect_prompt_image_paths(prompt_dir, file_format=file_format)
        if not prompt_images:
            continue

        start = cursor
        end = cursor + len(prompt_images)
        prompt_slices.append((prompt_name, prompt, start, end))
        image_paths.extend(prompt_images)
        texts.extend([prompt] * len(prompt_images))
        cursor = end

    if not image_paths:
        raise ValueError(f"No generated images found in {experiment_dir}")

    clip_values = clip_score(
        image_paths,
        texts,
        w=clip_weight,
        clip_model=clip_model,
        n_px=clip_image_size,
        batch_size=clip_batch_size,
        device=clip_device,
    )

    result: dict[str, tp.Any] = {"per_prompt": {}, "aggregate": {}}
    per_prompt_means: list[float] = []

    for prompt_name, prompt, start, end in prompt_slices:
        values = np.asarray(clip_values[start:end], dtype=np.float32)
        prompt_mean = float(values.mean())
        prompt_std = float(values.std())
        per_prompt_means.append(prompt_mean)
        result["per_prompt"][prompt_name] = {
            "prompt": prompt,
            "num_images": int(end - start),
            "clip_score_mean": prompt_mean,
            "clip_score_std": prompt_std,
        }

    fid_value = float(
        compute_fid(
            first_path=experiment_dir,
            first_fname=_generated_file_pattern(file_format),
            second_path=reference_dir,
            second_fname="*.jpg",
        )
    )

    per_prompt_means_arr = np.asarray(per_prompt_means, dtype=np.float32)
    result["aggregate"] = {
        "clip_score_mean": {
            "mean": float(per_prompt_means_arr.mean()),
            "std": float(per_prompt_means_arr.std()),
        },
        "fid": fid_value,
        "num_prompts": len(prompt_slices),
        "num_images": len(image_paths),
    }
    return result


def write_metrics_json(experiment_dir: str, metrics_payload: dict[str, tp.Any]) -> str:
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    save_json(metrics_path, metrics_payload)
    return metrics_path


def flatten_validation_metrics_for_summary(
    metrics_payload: dict[str, tp.Any],
    *,
    split: str,
    variant: str,
    strength: float | None,
    experiment_dir: str,
) -> dict[str, tp.Any]:
    aggregate = metrics_payload["aggregate"]
    return {
        "split": split,
        "variant": variant,
        "strength": strength,
        "clip_score_mean": float(aggregate["clip_score_mean"]["mean"]),
        "clip_score_std": float(aggregate["clip_score_mean"]["std"]),
        "fid": float(aggregate["fid"]),
        "num_prompts": int(aggregate["num_prompts"]),
        "num_images": int(aggregate["num_images"]),
        "experiment_dir": experiment_dir,
    }


def _dominates_validation(row_a: dict[str, tp.Any], row_b: dict[str, tp.Any]) -> bool:
    return (
        float(row_a["clip_score_mean"]) >= float(row_b["clip_score_mean"])
        and float(row_a["fid"]) <= float(row_b["fid"])
        and (
            float(row_a["clip_score_mean"]) > float(row_b["clip_score_mean"])
            or float(row_a["fid"]) < float(row_b["fid"])
        )
    )


def select_best_validation_strength_row(
    rows: tp.Sequence[dict[str, tp.Any]],
) -> tuple[dict[str, tp.Any], list[dict[str, tp.Any]]]:
    if not rows:
        raise ValueError("rows must not be empty")

    ranked = [dict(row) for row in rows]
    remaining = ranked[:]
    rank = 1
    while remaining:
        front = []
        for row in remaining:
            if not any(_dominates_validation(other, row) for other in remaining if other is not row):
                front.append(row)
        for row in front:
            row["pareto_rank"] = rank
            row["is_selected_best"] = False
        remaining = [row for row in remaining if row not in front]
        rank += 1

    clips = np.asarray([float(row["clip_score_mean"]) for row in ranked], dtype=np.float32)
    fids = np.asarray([float(row["fid"]) for row in ranked], dtype=np.float32)
    clip_min, clip_max = float(clips.min()), float(clips.max())
    fid_min, fid_max = float(fids.min()), float(fids.max())

    for row in ranked:
        clip_value = float(row["clip_score_mean"])
        fid_value = float(row["fid"])
        clip_norm = (clip_value - clip_min) / max(clip_max - clip_min, 1e-8)
        fid_norm = (fid_value - fid_min) / max(fid_max - fid_min, 1e-8)
        row["selection_score"] = float(clip_norm + (1.0 - fid_norm))

    pareto_rows = [row for row in ranked if row["pareto_rank"] == 1]
    best_row = max(
        pareto_rows,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["clip_score_mean"]),
            -float(row["fid"]),
        ),
    )
    for row in ranked:
        if row["experiment_dir"] == best_row["experiment_dir"]:
            row["is_selected_best"] = True
    return best_row, ranked


def write_summary_csv(rows: tp.Sequence[dict[str, tp.Any]], output_path: str) -> str:
    if not rows:
        raise ValueError("rows must not be empty")

    preferred = [
        "split",
        "variant",
        "strength",
        "clip_score_mean",
        "clip_score_std",
        "fid",
        "pareto_rank",
        "selection_score",
        "is_selected_best",
        "num_prompts",
        "num_images",
        "experiment_dir",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra

    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def strip_selection_fields_from_summary_rows(
    rows: tp.Sequence[dict[str, tp.Any]],
) -> list[dict[str, tp.Any]]:
    excluded = {
        "eligible_for_selection",
        "pareto_rank",
        "is_selected_best",
        "selection_score",
    }
    return [{key: value for key, value in row.items() if key not in excluded} for row in rows]


def save_validation_metric_plots(
    rows: tp.Sequence[dict[str, tp.Any]],
    output_dir: str,
) -> tuple[str | None, str | None]:
    steering_rows = [
        row for row in rows
        if row.get("variant") == "best_steering" and row.get("strength") is not None
    ]
    if not steering_rows:
        return None, None

    steering_rows = sorted(steering_rows, key=lambda row: float(row["strength"]))
    strengths = [float(row["strength"]) for row in steering_rows]
    clip_scores = [float(row["clip_score_mean"]) for row in steering_rows]
    fid_scores = [float(row["fid"]) for row in steering_rows]
    baseline_row = next((row for row in rows if row.get("variant") == "baseline"), None)

    ensure_dir(output_dir)

    clip_path = os.path.join(output_dir, "clip_vs_strength.png")
    plt.figure(figsize=(8, 5))
    plt.plot(strengths, clip_scores, marker="o")
    if baseline_row is not None:
        plt.axhline(
            float(baseline_row["clip_score_mean"]),
            linestyle="--",
            color="tab:gray",
            label="baseline",
        )
        plt.legend()
    plt.xlabel("strength")
    plt.ylabel("clip_score_mean")
    plt.title("CLIP vs strength")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(clip_path, dpi=150)
    plt.close()

    fid_path = os.path.join(output_dir, "fid_vs_strength.png")
    plt.figure(figsize=(8, 5))
    plt.plot(strengths, fid_scores, marker="o", color="tab:orange")
    if baseline_row is not None:
        plt.axhline(
            float(baseline_row["fid"]),
            linestyle="--",
            color="tab:gray",
            label="baseline",
        )
        plt.legend()
    plt.xlabel("strength")
    plt.ylabel("fid")
    plt.title("FID vs strength")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fid_path, dpi=150)
    plt.close()

    return clip_path, fid_path


def run_validation_strength_sweep(
    *,
    manifest: tp.Sequence[dict[str, tp.Any]],
    reference_dir: str,
    steering_source: str | dict,
    strengths: tp.Sequence[float],
    output_root: str,
    model_name: str,
    pipeline=None,
    device=None,
    split: str = "validation",
    file_format: str = "PNG",
    concept_seed: int = 0,
    include_baseline: bool = True,
    use_all_diffusion_steps: bool = False,
    renormalize_output: bool = True,
    clip_model: str = "ViT-B/32",
    clip_weight: float = 2.5,
    clip_image_size: int = 224,
    clip_batch_size: int = 50,
    clip_device: str | None = None,
) -> tuple[list[dict[str, tp.Any]], dict[str, tp.Any]]:
    if not strengths:
        raise ValueError("strengths must not be empty")

    device = device or get_device()
    own_pipeline = pipeline is None
    if own_pipeline:
        pipeline = init_pipeline_for_image_model(model=model_name)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)

    ensure_dir(output_root)
    summary_rows: list[dict[str, tp.Any]] = []

    if include_baseline:
        baseline_result = generate_dataset_variants(
            manifest=manifest,
            output_root=output_root,
            split=split,
            variants=[{"name": "baseline", "baseline": True}],
            model_name=model_name,
            pipeline=pipeline,
            device=device,
            file_format=file_format,
            concept_seed=concept_seed,
        )[0]
        baseline_metrics = evaluate_validation_experiment_dir(
            baseline_result["experiment_dir"],
            reference_dir,
            file_format=file_format,
            clip_model=clip_model,
            clip_weight=clip_weight,
            clip_image_size=clip_image_size,
            clip_batch_size=clip_batch_size,
            clip_device=clip_device,
        )
        write_metrics_json(baseline_result["experiment_dir"], baseline_metrics)
        baseline_row = flatten_validation_metrics_for_summary(
            baseline_metrics,
            split=split,
            variant="baseline",
            strength=None,
            experiment_dir=baseline_result["experiment_dir"],
        )
        baseline_row["eligible_for_selection"] = False
        summary_rows.append(baseline_row)

    steering_rows: list[dict[str, tp.Any]] = []
    for strength in strengths:
        variant_spec = {
            "name": "best_steering",
            "baseline": False,
            "steering_source": steering_source,
            "strength": float(strength),
            "use_all_diffusion_steps": use_all_diffusion_steps,
            "renormalize_output": renormalize_output,
            "concept_seed": concept_seed,
        }
        result = generate_dataset_variants(
            manifest=manifest,
            output_root=output_root,
            split=split,
            variants=[variant_spec],
            model_name=model_name,
            pipeline=pipeline,
            device=device,
            file_format=file_format,
            concept_seed=concept_seed,
        )[0]
        metrics_payload = evaluate_validation_experiment_dir(
            result["experiment_dir"],
            reference_dir,
            file_format=file_format,
            clip_model=clip_model,
            clip_weight=clip_weight,
            clip_image_size=clip_image_size,
            clip_batch_size=clip_batch_size,
            clip_device=clip_device,
        )
        write_metrics_json(result["experiment_dir"], metrics_payload)
        row = flatten_validation_metrics_for_summary(
            metrics_payload,
            split=split,
            variant="best_steering",
            strength=float(strength),
            experiment_dir=result["experiment_dir"],
        )
        row["eligible_for_selection"] = True
        steering_rows.append(row)

    best_row, annotated_steering_rows = select_best_validation_strength_row(steering_rows)
    summary_rows.extend(annotated_steering_rows)
    save_validation_metric_plots(summary_rows, output_root)

    public_summary_rows = strip_selection_fields_from_summary_rows(summary_rows)
    write_summary_csv(public_summary_rows, os.path.join(output_root, "summary.csv"))
    save_json(os.path.join(output_root, "summary.json"), public_summary_rows)
    save_json(os.path.join(output_root, "best_config.json"), best_row)
    return summary_rows, best_row


def generate_dataset_variants(
    *,
    manifest: tp.Sequence[dict[str, tp.Any]],
    output_root: str,
    split: str,
    variants: tp.Sequence[dict[str, tp.Any]],
    model_name: str,
    pipeline=None,
    device=None,
    file_format: str = "PNG",
    concept_seed: int = 0,
) -> list[dict[str, tp.Any]]:
    """Generate images for a manifest across multiple steering variants.

    Variant spec supports:
    - `name`: output label
    - `baseline`: if True, generate without steering
    - `steering_source`: path or loaded payload; either:
      - concept bank: `dict[concept_name][step][place][layer]`
      - steering vectors: `dict[step][place][layer]`
    - `strength`: additive steering strength
    - `randomize_source`: if True, derive a random vector with the same shape
    - `random_seed`: seed for random steering derivation
    - `use_all_diffusion_steps`: if False, step-0 vectors are reused for all steps
    - `renormalize_output`: preserve output norm after additive steering
    - `output_subdir`: optional output directory name override
    """
    if file_format not in EXTENSIONS:
        raise ValueError(f"Unsupported file_format={file_format!r}. Expected one of {sorted(EXTENSIONS)}")

    device = device or get_device()
    own_pipeline = pipeline is None
    if own_pipeline:
        pipeline = init_pipeline_for_image_model(model=model_name)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)

    ensure_dir(output_root)
    results: list[dict[str, tp.Any]] = []

    for variant in variants:
        variant_name = variant["name"]
        baseline = bool(variant.get("baseline", False))
        strength = variant.get("strength")
        use_all_diffusion_steps = bool(variant.get("use_all_diffusion_steps", False))
        renormalize_output = bool(variant.get("renormalize_output", True))
        variant_concept_seed = int(variant.get("concept_seed", concept_seed))
        output_subdir = variant.get("output_subdir") or build_variant_dir_name(split, variant_name, strength)
        experiment_dir = ensure_dir(os.path.join(output_root, output_subdir))

        steering_source = load_steering_source(
            variant.get("steering_source", variant.get("steering_path"))
        )
        random_source_path = None
        if variant.get("randomize_source", False):
            if steering_source is None:
                raise ValueError(f"Variant {variant_name!r} requested random steering without a steering source")
            random_seed = int(variant.get("random_seed", DEFAULT_RANDOM_STEERING_SEED))
            steering_source = create_random_steering_vectors(template_from_source(steering_source), seed=random_seed)
            random_source_path = os.path.join(experiment_dir, "random_steering_vector.pickle")
            save_pickle(steering_source, random_source_path)

        fixed_steering_vectors = steering_source if _is_steering_vectors(steering_source) else None
        concept_bank = steering_source if _is_concept_bank(steering_source) else None
        concept_names = sorted(concept_bank.keys()) if concept_bank else []

        config = {
            "split": split,
            "variant": variant_name,
            "model_name": model_name,
            "file_format": file_format,
            "baseline": baseline,
            "strength": strength,
            "use_all_diffusion_steps": use_all_diffusion_steps,
            "renormalize_output": renormalize_output,
            "concept_seed": variant_concept_seed,
            "steering_source_path": variant.get("steering_source") if isinstance(variant.get("steering_source"), str) else None,
            "random_steering_vector_path": random_source_path,
            "manifest_size": len(manifest),
        }
        save_json(os.path.join(experiment_dir, "config.json"), config)

        skipped = 0
        generated = 0
        shared_hook = None

        try:
            if not baseline and fixed_steering_vectors is not None:
                shared_hook = _hook_fixed_steering(
                    pipeline=pipeline,
                    model_name=model_name,
                    steering_vectors=fixed_steering_vectors,
                    strength=float(strength),
                    device=device,
                    use_all_diffusion_steps=use_all_diffusion_steps,
                    renormalize_output=renormalize_output,
                )

            print(f"Generating variant {variant_name} into {experiment_dir}")

            for record in manifest:
                prompt_dir = ensure_dir(_prompt_dir(experiment_dir, int(record["prompt_index"])))
                with open(os.path.join(prompt_dir, "prompt.txt"), "w", encoding="utf-8") as fout:
                    fout.write(record["caption"])

                if concept_bank is not None:
                    chosen_concepts: dict[str, str | None] = assign_concepts_to_seeds(
                        concept_names=concept_names,
                        prompt_index=int(record["prompt_index"]),
                        seeds=record["seeds"],
                        concept_seed=variant_concept_seed,
                    )
                else:
                    chosen_concepts = {}
                if _is_prompt_complete(experiment_dir, int(record["prompt_index"]), record["seeds"], file_format):
                    for seed in record["seeds"]:
                        if concept_bank is None:
                            chosen_concepts[str(seed)] = None
                    _save_prompt_metadata(prompt_dir, record, concept_names, chosen_concepts)
                    skipped += len(record["seeds"])
                    continue

                for seed in record["seeds"]:
                    image_path = _prompt_image_path(experiment_dir, int(record["prompt_index"]), int(seed), file_format)
                    if os.path.exists(image_path):
                        if concept_bank is None:
                            chosen_concepts[str(seed)] = None
                        skipped += 1
                        continue

                    steering_vectors = None
                    chosen_concept = None
                    if concept_bank is not None:
                        chosen_concept = tp.cast(str, chosen_concepts[str(seed)])
                        steering_vectors = concept_bank[chosen_concept]
                    elif fixed_steering_vectors is not None:
                        steering_vectors = fixed_steering_vectors

                    chosen_concepts[str(seed)] = chosen_concept
                    _generate_single_image(
                        pipeline=pipeline,
                        model_name=model_name,
                        prompt=record["caption"],
                        seed=int(seed),
                        device=device,
                        output_path=image_path,
                        file_format=file_format,
                        steering_vectors=None if baseline else steering_vectors,
                        strength=strength,
                        use_all_diffusion_steps=use_all_diffusion_steps,
                        renormalize_output=renormalize_output,
                        reuse_hook=shared_hook if (shared_hook is not None and concept_bank is None) else None,
                    )
                    generated += 1

                    if chosen_concept:
                        print(
                            f"[variant={variant_name}] [prompt={int(record['prompt_index']):04d}] "
                            f"seed={int(seed):02d} concept={chosen_concept} -> {image_path}"
                        )
                    else:
                        print(
                            f"[variant={variant_name}] [prompt={int(record['prompt_index']):04d}] "
                            f"seed={int(seed):02d} -> {image_path}"
                        )

                _save_prompt_metadata(prompt_dir, record, concept_names, chosen_concepts)
        finally:
            if shared_hook is not None:
                hook_manager, _ = shared_hook
                hook_manager.remove_hooks()

        results.append(
            {
                "split": split,
                "variant": variant_name,
                "experiment_dir": experiment_dir,
                "generated_images": generated,
                "skipped_images": skipped,
                "config_path": os.path.join(experiment_dir, "config.json"),
                "random_steering_vector_path": random_source_path,
            }
        )

        print(f"Variant {variant_name}: generated {generated}, skipped {skipped}")

    return results
