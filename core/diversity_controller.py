import abc
import logging
import typing as tp
import warnings
from collections import defaultdict
from typing import Any

import torch

logger = logging.getLogger()
EPS = 1e-6


class VectorControl(abc.ABC):
    def __init__(self, num_layers: int | None = None):
        self._active = True
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
        self.num_attn_layers = num_layers

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool):
        self._active = value

    def reset(self):
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)

    @abc.abstractmethod
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int):
        raise NotImplementedError

    def __call__(self, vector: torch.Tensor, place_in_unet: str):
        if not self.active:
            return vector

        block_index = self._current_position[place_in_unet]
        input_shape = vector.shape
        vector = self.forward(vector, self._diffusion_step, place_in_unet, block_index)
        assert vector.shape == input_shape
        self._current_position[place_in_unet] += 1

        self._current_attn_layer += 1
        if self._current_attn_layer == self.num_attn_layers:
            self._current_attn_layer = 0
            self._current_position = defaultdict(int)
            self._diffusion_step += 1
        return vector


SteeringVectors = tp.NewType("SteeringVectors", dict[int, dict[str, list[torch.Tensor]]])


class CrossAttentionOutputAdditiveSteering(VectorControl):
    """Additive steering over cross-attention outputs.

    The controller adds one or more steering vectors to the attention output and,
    optionally, rescales the result back to the original per-token norm.
    """

    def __init__(
        self,
        *,
        source_concepts: list[SteeringVectors],
        strength: float,
        device: Any,
        num_layers: int | None = None,
        use_first_diffusion_step: bool = False,
        renormalize_output: bool = True,
        output_dtype: torch.dtype | None = None,
    ):
        super().__init__(num_layers=num_layers)
        self.device = device
        self.strength = strength
        self.use_first_diffusion_step = use_first_diffusion_step
        self.renormalize_output = renormalize_output
        self.output_dtype = output_dtype

        if self.strength < 0:
            raise ValueError("Negative values of strength are not supported")

        self.casteer_vectors = []
        for source_concept in source_concepts:
            casteer_concept_vectors = defaultdict(lambda: defaultdict(list))
            for num_steer, place_payload in source_concept.items():
                for place_in_unet, layer_vectors in place_payload.items():
                    for steering_vector in layer_vectors:
                        if len(steering_vector.shape) == 1:
                            steering_vector = steering_vector.unsqueeze(0)
                        casteer_concept_vectors[num_steer][place_in_unet].append(
                            steering_vector.to(self.device)
                        )
            self.casteer_vectors.append(casteer_concept_vectors)

    def _resolve_step_key(self, casteer_vectors, diffusion_step: int) -> int | None:
        if not casteer_vectors:
            return None
        if self.use_first_diffusion_step:
            if 0 in casteer_vectors:
                return 0
            return sorted(casteer_vectors.keys())[0]
        if diffusion_step in casteer_vectors:
            return diffusion_step
        if 0 in casteer_vectors:
            return 0
        return sorted(casteer_vectors.keys())[0]

    def steer_additive(self, vector: torch.Tensor, steering_vector: torch.Tensor) -> torch.Tensor:
        assert len(vector.shape) == 4

        steering_vector = steering_vector.to(vector.device, dtype=vector.dtype)

        original_norm = None
        if self.renormalize_output:
            original_norm = torch.linalg.norm(vector, dim=-1, keepdim=True).clamp(min=EPS)

        steered_vector = vector + self.strength * steering_vector

        if self.renormalize_output:
            steered_norm = torch.linalg.norm(steered_vector, dim=-1, keepdim=True).clamp(min=EPS)
            steered_vector = (steered_vector / steered_norm) * original_norm

        return steered_vector

    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int):
        batch_size = vector.shape[0]
        if batch_size > 1:
            batch_slice = slice(batch_size // 2, None)
            warnings.warn(
                "Steering only the prompt part of classifier-free guidance "
                "(assumed the batch_idx=0 is not conditioned on the prompt)"
            )
        else:
            batch_slice = slice(None, None)

        vector = vector.detach().clone()

        if place_in_unet in ["up", "mid", "down", "joint", "single", "sana"]:
            for casteer_vectors in self.casteer_vectors:
                num_steer = self._resolve_step_key(casteer_vectors, diffusion_step)
                if num_steer is None:
                    continue
                place_vectors = casteer_vectors.get(num_steer, {}).get(place_in_unet)
                if not place_vectors or block_index >= len(place_vectors):
                    continue

                steering_vector = place_vectors[block_index]
                vector[batch_slice, ...] = self.steer_additive(
                    vector[batch_slice, ...],
                    steering_vector,
                )

        if self.output_dtype is not None:
            return vector.to(self.output_dtype)
        return vector
