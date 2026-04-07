import logging
import torch
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import enum

from .controller import (
    logger,
    VectorControl,
)


class DiffusionModelType(enum.StrEnum):
    SANA = 'sana'
    SD = 'sd'

    @staticmethod
    def from_model(model: str) -> 'DiffusionModelType':
        """Convert string to DiffusionModelType enum"""
        model = model.lower().strip()

        if model in ['sana', 'sana-sprint', 'sana-06', 'sana-sprint-06', 'sana15']:
            return DiffusionModelType.SANA
        elif model in ['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo']:
            return DiffusionModelType.SD
        else:
            raise ValueError(f"Unknown model type: {model}. Supported types: {list(DiffusionModelType)}")


class HookManager:
    """Manages registration and cleanup of hooks for cross-attention output steering."""

    def __init__(self, model_type: DiffusionModelType):
        self.hooks = []
        self.controls = []
        self.module_info = {}
        self.model_type = model_type

    def register_vector_controls_with_hooks(self, model, *controls: VectorControl):
        """Register vector controls using PyTorch hooks on cross-attention outputs."""
        self.controls = list(controls)
        self._clear_hooks()

        block_count = self._register_hooks_recursive(model)

        for control in self.controls:
            control.num_attn_layers = block_count

        return self.hooks

    def _register_hooks_recursive(self, model) -> int:
        """Recursively find transformer blocks and register hooks."""
        block_count = 0

        if self.model_type == DiffusionModelType.SANA:
            block_count += self._register_sana_model_hooks(model)
        elif self.model_type == DiffusionModelType.SD:
            for name, module in model.named_children():
                if "down" in name:
                    block_count += self._register_hooks_in_submodule(module, "down")
                elif "up" in name:
                    block_count += self._register_hooks_in_submodule(module, "up")
                elif "mid" in name:
                    block_count += self._register_hooks_in_submodule(module, "mid")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return block_count

    def _register_sana_model_hooks(self, model) -> int:
        """Register hooks for SANA model architecture."""
        block_count = 0

        if hasattr(model, 'blocks'):
            for block in model.blocks:
                self._register_hooks_for_sana_block(block, "sana")
                block_count += 1

        if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            for block in model.transformer.blocks:
                self._register_hooks_for_sana_block(block, "sana")
                block_count += 1

        # Fallback: search recursively
        if block_count == 0:
            for name, module in model.named_modules():
                class_name = module.__class__.__name__
                if any(sana_name in class_name for sana_name in ['SanaTransformerBlock']):
                    self._register_hooks_for_sana_block(module, "sana")
                    block_count += 1

        return block_count

    def _register_hooks_in_submodule(self, module, place_in_unet: str) -> int:
        """Register hooks in a submodule for a specific place in UNet (SD architecture)."""
        block_count = 0

        for name, submodule in module.named_modules():
            class_name = submodule.__class__.__name__

            if class_name == 'BasicTransformerBlock':
                self._register_hooks_for_sd_block(submodule, place_in_unet)
                block_count += 1
            elif any(sana_name in class_name for sana_name in ['SanaTransformerBlock']):
                self._register_hooks_for_sana_block(submodule, place_in_unet)
                block_count += 1

        return block_count

    def _register_hooks_for_sd_block(self, block, place_in_unet: str):
        """Register cross-attention output hook for a Stable Diffusion BasicTransformerBlock."""
        self.module_info[id(block)] = place_in_unet

        for control in self.controls:
            # Hook into the cross-attention output (attn2)
            if hasattr(block, 'attn2') and block.attn2 is not None:
                hook = block.attn2.register_forward_hook(
                    self._create_attn_output_hook(control, place_in_unet)
                )
                self.hooks.append(hook)

    def _register_hooks_for_sana_block(self, block, place_in_unet: str):
        """Register cross-attention output hook for a SANA transformer block."""
        self.module_info[id(block)] = place_in_unet

        for control in self.controls:
            attn_module = self._get_sana_attention_module(block)
            if attn_module is not None:
                hook = attn_module.register_forward_hook(
                    self._create_sana_attn_output_hook(control, place_in_unet)
                )
                self.hooks.append(hook)

    def _get_sana_attention_module(self, block):
        """Get the appropriate attention module from a SANA block."""
        possible_names = ['attn2', 'attention', 'self_attn', 'Attention']

        for name in possible_names:
            if hasattr(block, name):
                attn_module = getattr(block, name)
                if attn_module is not None and hasattr(attn_module, 'to_q'):
                    return attn_module

        return None

    def _create_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for SD cross-attention output."""
        def hook_fn(module, input, output):
            if not control.active:
                return output

            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            return controlled_output[..., 0, :]

        return hook_fn

    def _create_sana_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for SANA cross-attention output."""
        def hook_fn(module, input, output):
            if not control.active:
                return output

            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            controlled_output = controlled_output[..., 0, :].to(torch.bfloat16)

            return controlled_output

        return hook_fn

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_info.clear()

    def remove_hooks(self):
        """Public method to remove all hooks."""
        self._clear_hooks()

    def reset_controls(self):
        """Reset all controls to initial state."""
        for control in self.controls:
            control.reset()


def diffusion_register_vector_controls_with_hooks(model, *controls: VectorControl, model_type: DiffusionModelType) -> HookManager:
    """
    Register vector controls using PyTorch hooks on cross-attention outputs.

    Args:
        model: The model to register controls on
        *controls: VectorControl instances to register
        model_type: The type of diffusion model

    Returns:
        HookManager: Manager object that can be used to remove hooks later
    """
    manager = HookManager(model_type=model_type)
    manager.register_vector_controls_with_hooks(model, *controls)
    return manager
