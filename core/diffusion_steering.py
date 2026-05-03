import enum

import torch

from .controller import VectorControl


class DiffusionModelType(enum.StrEnum):
    SANA = "sana"
    SD = "sd"

    @staticmethod
    def from_model(model: str) -> "DiffusionModelType":
        model = model.lower().strip()

        if model in ["sana", "sana-sprint", "sana-06", "sana-sprint-06", "sana15"]:
            return DiffusionModelType.SANA
        if model in ["sd14", "sd21", "sd21-turbo", "sdxl", "sdxl-turbo"]:
            return DiffusionModelType.SD
        raise ValueError(f"Unknown model type: {model}. Supported types: {list(DiffusionModelType)}")


class HookManager:
    """Manages registration and cleanup of hooks for vector controls."""

    def __init__(self, model_type: DiffusionModelType):
        self.hooks = []
        self.controls = []
        self.module_info = {}
        self.model_type = model_type

    def register_vector_controls_with_hooks(self, model, *controls: VectorControl):
        self.controls = list(controls)
        self._clear_hooks()

        block_count = self._register_hooks_recursive(model)
        for control in self.controls:
            control.num_attn_layers = block_count
        return self.hooks

    def _register_hooks_recursive(self, model) -> int:
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
        block_count = 0

        if hasattr(model, "blocks"):
            for block in model.blocks:
                self._register_hooks_for_sana_block(block, "sana")
                block_count += 1

        if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
            for block in model.transformer.blocks:
                self._register_hooks_for_sana_block(block, "sana")
                block_count += 1

        if block_count == 0:
            for _, module in model.named_modules():
                class_name = module.__class__.__name__
                if "SanaTransformerBlock" in class_name:
                    self._register_hooks_for_sana_block(module, "sana")
                    block_count += 1

        return block_count

    def _register_hooks_in_submodule(self, module, place_in_unet: str) -> int:
        block_count = 0
        for _, submodule in module.named_modules():
            class_name = submodule.__class__.__name__

            if class_name == "BasicTransformerBlock":
                self._register_hooks_for_sd_block(submodule, place_in_unet)
                block_count += 1
            elif "SanaTransformerBlock" in class_name:
                self._register_hooks_for_sana_block(submodule, place_in_unet)
                block_count += 1
        return block_count

    def _register_hooks_for_sd_block(self, block, place_in_unet: str):
        self.module_info[id(block)] = place_in_unet

        for control in self.controls:
            if hasattr(block, "attn2") and block.attn2 is not None:
                hook = block.attn2.register_forward_hook(
                    self._create_attn_output_hook(control, place_in_unet)
                )
                self.hooks.append(hook)

    def _register_hooks_for_sana_block(self, block, place_in_unet: str):
        self.module_info[id(block)] = place_in_unet

        for control in self.controls:
            attn_module = self._get_sana_attention_module(block)
            if attn_module is not None:
                hook = attn_module.register_forward_hook(
                    self._create_sana_attn_output_hook(control, place_in_unet)
                )
                self.hooks.append(hook)

    def _get_sana_attention_module(self, block):
        for name in ["attn2", "attention", "self_attn", "Attention"]:
            if hasattr(block, name):
                attn_module = getattr(block, name)
                if attn_module is not None and hasattr(attn_module, "to_q"):
                    return attn_module
        return None

    def _create_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        def hook_fn(module, inputs, output):
            del module, inputs
            if not control.active:
                return output
            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            return controlled_output[..., 0, :]

        return hook_fn

    def _create_sana_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        def hook_fn(module, inputs, output):
            del module, inputs
            if not control.active:
                return output
            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            return controlled_output[..., 0, :].to(torch.bfloat16)

        return hook_fn

    def _clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_info.clear()

    def remove_hooks(self):
        self._clear_hooks()

    def reset_controls(self):
        for control in self.controls:
            control.reset()


def diffusion_register_vector_controls_with_hooks(
    model,
    *controls: VectorControl,
    model_type: DiffusionModelType,
) -> HookManager:
    manager = HookManager(model_type=model_type)
    manager.register_vector_controls_with_hooks(model, *controls)
    return manager
