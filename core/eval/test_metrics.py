from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..create_dataset import ensure_dir, load_json, save_json
from .clip import get_clip_preprocess

SUPPORTED_TEST_METRICS = (
    "clip_score",
    "pick_score",
    "image_reward",
    "mps",
    "vendi",
)

DEFAULT_PICKSCORE_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
DEFAULT_PICKSCORE_MODEL = "yuvalkirstain/PickScore_v1"
DEFAULT_IMAGE_REWARD_MODEL = "ImageReward-v1.0"
DEFAULT_MPS_MODEL = "RE-N-Y/mpsv1"
DEFAULT_VARIANT_ORDER = ("baseline", "random_steering", "best_steering")
IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "webp")


def resolve_metric_device(device: str | None = None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _image_paths_for_prompt(prompt_dir: str, file_format: str | None = None) -> list[str]:
    if file_format is not None:
        ext = file_format.lower()
        if ext == "jpeg":
            ext = "jpg"
        return sorted(glob.glob(os.path.join(prompt_dir, f"*.{ext}")))

    paths: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(prompt_dir, f"*.{ext}")))
    return sorted(paths)


def _load_rgb_images(image_paths: Sequence[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            images.append(img.convert("RGB"))
    return images


def _scores_to_numpy(scores: Any) -> np.ndarray:
    if torch.is_tensor(scores):
        arr = scores.detach().cpu().numpy()
    else:
        arr = np.asarray(scores)
    return arr.astype(np.float32).reshape(-1)


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / embeddings.norm(dim=-1, p=2, keepdim=True).clamp_min(1e-12)


@dataclass(frozen=True)
class PromptRecord:
    prompt_name: str
    prompt: str
    image_paths: list[str]


def load_prompt_records(experiment_dir: str, file_format: str | None = None) -> list[PromptRecord]:
    prompt_dirs = sorted(glob.glob(os.path.join(experiment_dir, "prompt_*")))
    records: list[PromptRecord] = []

    for prompt_dir in prompt_dirs:
        prompt_path = os.path.join(prompt_dir, "prompt.txt")
        if not os.path.exists(prompt_path):
            continue
        with open(prompt_path, "r", encoding="utf-8") as fin:
            prompt = fin.read().strip()
        image_paths = _image_paths_for_prompt(prompt_dir, file_format=file_format)
        if not image_paths:
            continue
        records.append(
            PromptRecord(
                prompt_name=os.path.basename(prompt_dir),
                prompt=prompt,
                image_paths=image_paths,
            )
        )

    if not records:
        raise ValueError(f"No prompt/image records found in {experiment_dir}")
    return records


class CLIPBackbone:
    def __init__(
        self,
        *,
        clip_model: str = "ViT-B/32",
        image_size: int = 224,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        import clip

        self.device = resolve_metric_device(device)
        self.batch_size = batch_size
        self.model, _ = clip.load(clip_model, device=self.device)
        self.model.eval()
        self.image_preprocess, self.text_preprocess = get_clip_preprocess(image_size)

    @torch.no_grad()
    def _encode_image_batch(self, image_paths: Sequence[str]) -> torch.Tensor:
        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                tensors.append(self.image_preprocess(img))
        pixel_values = torch.stack(tensors, dim=0).to(self.device)
        image_features = self.model.encode_image(pixel_values)
        return image_features / image_features.norm(dim=-1, p=2, keepdim=True)

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[str]) -> torch.Tensor:
        features = []
        for batch_paths in _batched(list(image_paths), self.batch_size):
            features.append(self._encode_image_batch(batch_paths).cpu())
        return torch.cat(features, dim=0)

    @torch.no_grad()
    def score_prompt(self, image_paths: Sequence[str], prompt: str, weight: float = 2.5) -> np.ndarray:
        text_inputs = self.text_preprocess([prompt]).to(self.device)
        text_features = self.model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, p=2, keepdim=True)

        scores = []
        for batch_paths in _batched(list(image_paths), self.batch_size):
            image_features = self._encode_image_batch(batch_paths)
            batch_scores = weight * (image_features * text_features).sum(dim=-1)
            scores.append(batch_scores.clamp(min=0).cpu())
        return _scores_to_numpy(torch.cat(scores, dim=0))


class PickScoreMetric:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_PICKSCORE_MODEL,
        processor_name: str = DEFAULT_PICKSCORE_PROCESSOR,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = resolve_metric_device(device)
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

    @torch.no_grad()
    def score_prompt(self, image_paths: Sequence[str], prompt: str) -> np.ndarray:
        scores = []
        for batch_paths in _batched(list(image_paths), self.batch_size):
            images = _load_rgb_images(batch_paths)
            image_inputs = self.processor(
                images=images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = self.processor(
                text=[prompt] * len(batch_paths),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}
            text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

            # PickScore's official inference code targets transformers==4.27.3,
            # where get_*_features returns projected embedding tensors directly.
            image_embs = _normalize_embeddings(self.model.get_image_features(**image_inputs))
            text_embs = _normalize_embeddings(self.model.get_text_features(**text_inputs))

            batch_scores = self.model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
            scores.append(batch_scores.cpu())
        return _scores_to_numpy(torch.cat(scores, dim=0))


class ImageRewardMetric:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_IMAGE_REWARD_MODEL,
        device: str | None = None,
    ) -> None:
        import ImageReward as RM

        self.device = resolve_metric_device(device)
        self.model = RM.load(model_name)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    @torch.no_grad()
    def score_prompt(self, image_paths: Sequence[str], prompt: str) -> np.ndarray:
        scores = self.model.score(prompt, list(image_paths))
        return _scores_to_numpy(scores)


class MPSMetric:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MPS_MODEL,
        device: str | None = None,
    ) -> None:
        from imscore.mps.model import MPS

        self.device = resolve_metric_device(device)
        self.model = MPS.from_pretrained(model_name).eval().to(self.device)

    @torch.no_grad()
    def score_prompt(self, image_paths: Sequence[str], prompt: str) -> np.ndarray:
        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                image = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(image).permute(2, 0, 1))

        unique_shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(unique_shapes) == 1:
            pixels = torch.stack(tensors, dim=0).to(self.device)
            scores = self.model.score(pixels, [prompt] * len(image_paths))
            return _scores_to_numpy(scores)

        scores = []
        for tensor in tensors:
            value = self.model.score(tensor.unsqueeze(0).to(self.device), [prompt])
            scores.append(_scores_to_numpy(value)[0])
        return np.asarray(scores, dtype=np.float32)


def vendi_score_from_embeddings(embeddings: np.ndarray) -> float:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_images, dim]")

    num_images = embeddings.shape[0]
    if num_images == 0:
        raise ValueError("Cannot compute Vendi score for an empty set of embeddings")
    if num_images == 1:
        return 1.0

    normalized = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
    similarity = normalized @ normalized.T
    similarity = 0.5 * (similarity + similarity.T)

    eigenvalues = np.linalg.eigvalsh(similarity / float(num_images))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0:
        return 0.0
    eigenvalues = eigenvalues / total
    nonzero = eigenvalues[eigenvalues > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    return float(np.exp(entropy))


class TestMetricsEvaluator:
    def __init__(
        self,
        *,
        metrics: Sequence[str] = SUPPORTED_TEST_METRICS,
        device: str | None = None,
        clip_model: str = "ViT-B/32",
        clip_weight: float = 2.5,
        clip_image_size: int = 224,
        batch_size: int = 16,
        pickscore_model: str = DEFAULT_PICKSCORE_MODEL,
        pickscore_processor: str = DEFAULT_PICKSCORE_PROCESSOR,
        image_reward_model: str = DEFAULT_IMAGE_REWARD_MODEL,
        mps_model: str = DEFAULT_MPS_MODEL,
    ) -> None:
        metrics = tuple(metrics)
        unknown = sorted(set(metrics) - set(SUPPORTED_TEST_METRICS))
        if unknown:
            raise ValueError(f"Unknown metrics requested: {unknown}")

        self.metrics = metrics
        self.device = resolve_metric_device(device)
        self.clip_model_name = clip_model
        self.clip_weight = clip_weight
        self.clip_image_size = clip_image_size
        self.batch_size = batch_size
        self.pickscore_model_name = pickscore_model
        self.pickscore_processor_name = pickscore_processor
        self.image_reward_model_name = image_reward_model
        self.mps_model_name = mps_model

        self._clip_backend: CLIPBackbone | None = None
        self._pickscore_backend: PickScoreMetric | None = None
        self._image_reward_backend: ImageRewardMetric | None = None
        self._mps_backend: MPSMetric | None = None

    def _clip_backend_instance(self) -> CLIPBackbone:
        if self._clip_backend is None:
            self._clip_backend = CLIPBackbone(
                clip_model=self.clip_model_name,
                image_size=self.clip_image_size,
                batch_size=self.batch_size,
                device=self.device,
            )
        return self._clip_backend

    def _pickscore_backend_instance(self) -> PickScoreMetric:
        if self._pickscore_backend is None:
            self._pickscore_backend = PickScoreMetric(
                model_name=self.pickscore_model_name,
                processor_name=self.pickscore_processor_name,
                batch_size=self.batch_size,
                device=self.device,
            )
        return self._pickscore_backend

    def _image_reward_backend_instance(self) -> ImageRewardMetric:
        if self._image_reward_backend is None:
            self._image_reward_backend = ImageRewardMetric(
                model_name=self.image_reward_model_name,
                device=self.device,
            )
        return self._image_reward_backend

    def _mps_backend_instance(self) -> MPSMetric:
        if self._mps_backend is None:
            self._mps_backend = MPSMetric(
                model_name=self.mps_model_name,
                device=self.device,
            )
        return self._mps_backend

    def evaluate_experiment_dir(
        self,
        experiment_dir: str,
        *,
        file_format: str | None = None,
    ) -> dict[str, Any]:
        prompt_records = load_prompt_records(experiment_dir, file_format=file_format)

        result: dict[str, Any] = {
            "metadata": {
                "metrics": list(self.metrics),
                "device": self.device,
                "clip_model": self.clip_model_name,
                "clip_weight": self.clip_weight,
                "clip_image_size": self.clip_image_size,
                "pickscore_model": self.pickscore_model_name,
                "pickscore_processor": self.pickscore_processor_name,
                "image_reward_model": self.image_reward_model_name,
                "mps_model": self.mps_model_name,
                "vendi_embedding_model": self.clip_model_name if "vendi" in self.metrics else None,
            },
            "per_prompt": {},
            "aggregate": {},
        }
        aggregate_values: dict[str, list[float]] = {metric: [] for metric in self.metrics}
        num_images = 0

        for record in tqdm(prompt_records, desc=f"Evaluating {os.path.basename(experiment_dir)}"):
            prompt_payload: dict[str, Any] = {
                "prompt": record.prompt,
                "num_images": len(record.image_paths),
            }

            if "clip_score" in self.metrics:
                clip_values = self._clip_backend_instance().score_prompt(
                    record.image_paths,
                    record.prompt,
                    weight=self.clip_weight,
                )
                prompt_payload["clip_score_mean"] = float(np.mean(clip_values))
                prompt_payload["clip_score_std"] = float(np.std(clip_values))
                aggregate_values["clip_score"].append(prompt_payload["clip_score_mean"])

            if "pick_score" in self.metrics:
                pick_values = self._pickscore_backend_instance().score_prompt(record.image_paths, record.prompt)
                prompt_payload["pick_score_mean"] = float(np.mean(pick_values))
                prompt_payload["pick_score_std"] = float(np.std(pick_values))
                aggregate_values["pick_score"].append(prompt_payload["pick_score_mean"])

            if "image_reward" in self.metrics:
                reward_values = self._image_reward_backend_instance().score_prompt(record.image_paths, record.prompt)
                prompt_payload["image_reward_mean"] = float(np.mean(reward_values))
                prompt_payload["image_reward_std"] = float(np.std(reward_values))
                aggregate_values["image_reward"].append(prompt_payload["image_reward_mean"])

            if "mps" in self.metrics:
                mps_values = self._mps_backend_instance().score_prompt(record.image_paths, record.prompt)
                prompt_payload["mps_mean"] = float(np.mean(mps_values))
                prompt_payload["mps_std"] = float(np.std(mps_values))
                aggregate_values["mps"].append(prompt_payload["mps_mean"])

            if "vendi" in self.metrics:
                embeddings = self._clip_backend_instance().encode_images(record.image_paths).numpy()
                vendi_value = vendi_score_from_embeddings(embeddings)
                prompt_payload["vendi"] = vendi_value
                aggregate_values["vendi"].append(vendi_value)

            result["per_prompt"][record.prompt_name] = prompt_payload
            num_images += len(record.image_paths)

        aggregate: dict[str, Any] = {
            "num_prompts": len(prompt_records),
            "num_images": num_images,
        }
        for metric, values in aggregate_values.items():
            values_arr = np.asarray(values, dtype=np.float32)
            aggregate[metric] = {
                "mean": float(values_arr.mean()),
                "std": float(values_arr.std()),
            }

        result["aggregate"] = aggregate
        return result


def write_metrics_json(
    experiment_dir: str,
    metrics_payload: dict[str, Any],
    filename: str = "test_metrics.json",
) -> str:
    metrics_path = os.path.join(experiment_dir, filename)
    save_json(metrics_path, metrics_payload)
    return metrics_path


def flatten_test_metrics_for_summary(
    metrics_payload: dict[str, Any],
    *,
    config: dict[str, Any],
    experiment_dir: str,
) -> dict[str, Any]:
    aggregate = metrics_payload["aggregate"]
    row: dict[str, Any] = {
        "split": config.get("split"),
        "variant": config.get("variant"),
        "strength": config.get("strength"),
        "model_name": config.get("model_name"),
        "file_format": config.get("file_format"),
        "num_prompts": int(aggregate["num_prompts"]),
        "num_images": int(aggregate["num_images"]),
        "experiment_dir": experiment_dir,
    }

    for metric in metrics_payload["metadata"]["metrics"]:
        row[f"{metric}_mean"] = float(aggregate[metric]["mean"])
        row[f"{metric}_std"] = float(aggregate[metric]["std"])
    return row


def add_baseline_deltas(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows]
    baseline_row = next((row for row in rows if row.get("variant") == "baseline"), None)
    if baseline_row is None:
        return rows

    for row in rows:
        for metric in metrics:
            key = f"{metric}_mean"
            if key in row and key in baseline_row:
                row[f"delta_vs_baseline_{metric}"] = float(row[key]) - float(baseline_row[key])
    return rows


def write_summary_csv(rows: Sequence[dict[str, Any]], output_path: str) -> str:
    if not rows:
        raise ValueError("rows must not be empty")

    preferred = [
        "split",
        "variant",
        "strength",
        "model_name",
        "file_format",
        "num_prompts",
        "num_images",
        "experiment_dir",
    ]
    metric_keys = sorted(
        key
        for key in {column for row in rows for column in row.keys()}
        if key not in preferred
    )
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + metric_keys

    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def discover_experiment_dirs(
    output_root: str,
    *,
    split: str | None = None,
    variants: Sequence[str] | None = None,
    variant_order: Sequence[str] = DEFAULT_VARIANT_ORDER,
) -> list[dict[str, Any]]:
    variant_filter = set(variants) if variants is not None else None
    rank = {name: idx for idx, name in enumerate(variant_order)}
    discovered: list[dict[str, Any]] = []

    for name in sorted(os.listdir(output_root)):
        experiment_dir = os.path.join(output_root, name)
        if not os.path.isdir(experiment_dir):
            continue
        config_path = os.path.join(experiment_dir, "config.json")
        if not os.path.exists(config_path):
            continue

        config = load_json(config_path)
        if split is not None and config.get("split") != split:
            continue
        if variant_filter is not None and config.get("variant") not in variant_filter:
            continue

        discovered.append(
            {
                "experiment_dir": experiment_dir,
                "config": config,
            }
        )

    discovered.sort(
        key=lambda item: (
            rank.get(str(item["config"].get("variant")), len(rank)),
            -1.0 if item["config"].get("strength") is None else float(item["config"]["strength"]),
            item["experiment_dir"],
        )
    )
    return discovered
