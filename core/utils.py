import os
import torch

from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image

try:
    from diffusers import SanaPipeline
except ImportError:
    try:
        from diffusers.pipelines.sana import SanaPipeline
    except ImportError:
        try:
            import sys
            possible_sana_paths = ['./Sana', '../Sana', './sana', '../sana']
            for path in possible_sana_paths:
                if os.path.exists(path):
                    sys.path.insert(0, path)
                    break
            from app.sana_pipeline import SanaPipeline
        except ImportError:
            SanaPipeline = None
            print("Warning: SANA is not available. Install SANA or ensure diffusers supports SanaPipeline.")

try:
    from diffusers import SanaSprintPipeline
except ImportError:
    SanaSprintPipeline = None
    print("Warning: SANA-Sprint is not available.")


def _get_hf_token():
    """Get HuggingFace token from environment variable."""
    return os.environ.get('HF_TOKEN', None)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


SUPPORTED_DIFFUSION_MODELS = [
    'sd14', 'sd21', 'sd21-turbo',
    'sdxl', 'sdxl-turbo',
    'sana', 'sana15', 'sana-sprint',
    'sana-06', 'sana-sprint-06',
]


def init_pipeline_for_image_model(model: str) -> DiffusionPipeline:
    token = _get_hf_token()
    if model == 'sd14':
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            cache_dir='./cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'sd21':
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            cache_dir='./cache',
            device_map='balanced',
        )
    elif model == 'sd21-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir='./cache',
            device_map='balanced',
        )
    elif model == 'sdxl':
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir='./cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'sdxl-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir='./cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'sana15':
        if SanaPipeline is None:
            raise ValueError("SANA is not available.")
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16,
            cache_dir='./cache',
            token=token,
            device_map='balanced',
        )
    elif model == 'sana':
        if SanaPipeline is None:
            raise ValueError("SANA is not available.")
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_teacher_diffusers",
            torch_dtype=torch.bfloat16,
            cache_dir='./cache',
            token=token,
            device_map='balanced',
        )
    elif model == 'sana-06':
        if SanaPipeline is None:
            raise ValueError("SANA is not available.")
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers",
            torch_dtype=torch.bfloat16,
            cache_dir='./cache',
            token=token,
            device_map='balanced',
        )
    elif model == 'sana-sprint':
        if SanaSprintPipeline is None:
            raise ValueError("SANA-Sprint is not available.")
        pipe = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16,
            cache_dir='./cache',
            token=token,
            device_map='balanced',
        )
    elif model == 'sana-sprint-06':
        if SanaSprintPipeline is None:
            raise ValueError("SANA-Sprint is not available.")
        pipe = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16,
            cache_dir='./cache',
            token=token,
            device_map='balanced',
        )
    else:
        raise ValueError(f'Unknown model: {model}')
    return pipe


def get_num_denoising_steps(model: str) -> int:
    if model in ('sd14', 'sd21'):
        return 50
    elif model in ('sd21-turbo', 'sdxl-turbo'):
        return 1
    elif model in ('sdxl',):
        return 30
    elif model in ('sana', 'sana-06', 'sana15'):
        return 20
    elif model in ('sana-sprint', 'sana-sprint-06'):
        return 1
    else:
        raise ValueError(f'Unknown model type: {model}')


def run_image_model(model_type: str, pipe, prompt: str, seed: int, device: torch.device, num_images: int = 1):
    if model_type in ['sd14', 'sd21', 'sdxl']:
        images = pipe(
            prompt=prompt,
            num_inference_steps=get_num_denoising_steps(model_type),
            generator=torch.Generator(device=device).manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    elif model_type in ['sd21-turbo', 'sdxl-turbo']:
        images = pipe(
            prompt=prompt,
            num_inference_steps=get_num_denoising_steps(model_type),
            guidance_scale=0.0,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    elif model_type in ['sana', 'sana-06', 'sana15']:
        images = pipe(
            prompt=prompt,
            num_inference_steps=get_num_denoising_steps(model_type),
            height=1024,
            width=1024,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    elif model_type in ['sana-sprint', 'sana-sprint-06']:
        images = pipe(
            prompt=prompt,
            num_inference_steps=get_num_denoising_steps(model_type),
            height=1024,
            width=1024,
            intermediate_timesteps=None,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    return images
