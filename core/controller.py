import logging
import warnings
import torch
import abc
import typing as tp
from collections import defaultdict
from typing import Any

logger = logging.getLogger()

EPS = 1e-6


class VectorControl(abc.ABC):
    def __init__(self, num_layers: int = None):
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

# For each diffusion step,
# for each place in the network represented as string key,
# for each layer position, we store steering vector
SteeringVectors = tp.NewType('SteeringVectors', dict[int, dict[str, list[torch.Tensor]]])


class CrossAttentionOutputSteering(VectorControl):
    """
    CASteer: Cross-Attention Output Steering.

    Implements concept erasure and concept switching by projecting out
    the steering direction from cross-attention outputs.

    Equation 6 (with intermediate clipping):
        alpha = max(beta * <ca_X, ca_out>, 0)
        ca_out_new = ca_out - alpha * ca_X

    Equation 5 (matrix form):
        s_new = (I - beta * s * s^T) * c
    """

    def __init__(
        self,
        *,
        source_concepts: list[SteeringVectors],
        target_concepts: list[SteeringVectors | None],
        strength: float,
        device: Any,
        num_layers: int = None,
        intermediate_clipping: bool = True,
        use_first_diffusion_step: bool = False,
    ):
        super().__init__(num_layers=num_layers)
        self.device = device
        self.intermediate_clipping = intermediate_clipping
        self.strength = strength
        self.use_first_diffusion_step = use_first_diffusion_step

        if self.strength < 0:
            raise ValueError('Negative values of strength are not supported')

        self.casteer_vectors = []
        for source_concept, target_concept in zip(source_concepts, target_concepts):
            casteer_concept_transforms = defaultdict(lambda: defaultdict(list))
            for num_steer in source_concept:
                for place_in_unet in source_concept[num_steer]:
                    for block_idx in range(len(source_concept[num_steer][place_in_unet])):
                        source_vector = source_concept[num_steer][place_in_unet][block_idx]
                        if target_concept is not None:
                            target_vector = target_concept[num_steer][place_in_unet][block_idx]
                        else:
                            target_vector = torch.zeros_like(source_vector)
                        steering_vector = source_vector - target_vector

                        if len(steering_vector.shape) == 1:
                            steering_vector = steering_vector.unsqueeze(0)
                        steering_vector = steering_vector.to(self.device).unsqueeze(-1)

                        # Precompute projection matrix P = I - beta * v * v^+ (Eq. 5)
                        res = self.strength * (steering_vector @ torch.linalg.pinv(steering_vector))
                        P = torch.eye(res.shape[1], dtype=res.dtype, device=self.device).unsqueeze(0) - res

                        casteer_concept_transforms[num_steer][place_in_unet].append((steering_vector.squeeze(-1), P))
            self.casteer_vectors.append(casteer_concept_transforms)

        self.steering_cache = {}

    def steer_matrix_form(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        """Apply steering using precomputed projection matrix (Eq. 5)."""
        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]
        (_, P) = steering_tensors

        vector_steered = ((
            vector.to(self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1) @ P.to(vector.device).mT
        )).transpose(0, 1).reshape(batch_size, sequence_length, num_heads, hidden_dim)
        return vector_steered

    def steer_with_clipping(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        """
        Apply steering with intermediate clipping (Eq. 6):
            alpha = max(beta * <ca_X, ca_out>, 0)
            ca_out_new = ca_out - alpha * ca_X
        """
        assert len(vector.shape) == 4

        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]
        (b, _) = steering_tensors

        b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)

        vector_reshaped = vector.to(self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
        b_norm_reshaped = b_norm.unsqueeze(-1)

        # Compute dot products between vector components and steering vector
        projection_scores = (
            vector_reshaped @ b_norm_reshaped
        ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)

        # Clip: only steer when dot product is positive (concept is present)
        if self.intermediate_clipping:
            projection_scores = torch.where(projection_scores > 0, projection_scores, 0)

        steering_delta = -self.strength * projection_scores.to(vector.device) * b_norm.to(vector.device)

        return vector + steering_delta

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int):
        batch_size = vector.shape[0]
        if batch_size > 1:
            # Steer only the prompt part of classifier-free guidance
            batch_slice = slice(batch_size // 2, None)
            warnings.warn('Steering only the prompt part of classifier-free guidance (assumed the batch_idx=0 is not conditioned on the prompt)')
        else:
            batch_slice = slice(None, None)

        vector = vector.detach().clone()

        if place_in_unet in ['up', 'mid', 'down', 'joint', 'single', 'sana']:
            # Use first step vectors for all steps (turbo/sprint) or per-step vectors
            num_steer = 0 if self.use_first_diffusion_step else diffusion_step

            for casteer_vectors in self.casteer_vectors:
                vector[batch_slice, ...] = self.steer_with_clipping(
                    vector[batch_slice, ...],
                    *casteer_vectors[num_steer][place_in_unet][block_index]
                )
        return vector.half()
