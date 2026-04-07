import argparse
import json
import os
import math
import typing as tp

import pandas as pd
from diffusers import DiffusionPipeline

from core.controller import CrossAttentionOutputSteering, VectorControl
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.pickle import unpickle
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, run_image_model

SAVE_OPTIONS = {
    'PNG': {},
    'JPEG': {
        'subsampling': '4:4:4',
        'quality': 95,
    },
}

EXTENSIONS = {
    'PNG': 'png',
    'JPEG': 'jpg',
}


def load_prompts(args):
    """Load evaluation prompts: either from ImageNet templates or COCO captions."""
    if args.generate_concept != 'coco':
        template_path = args.template_path or 'exp/datasets/eval/imagenet/template.json'
        with open(template_path) as f:
            templates = json.load(f)
        return [t.format(args.generate_concept) for t in templates], args.num_images_per_prompt
    else:
        df = pd.read_csv('exp/datasets/eval/coco/coco_30k.csv')
        prompts = [p for p in df['prompt'] if 'horse' not in p.lower()]
        if args.max_samples:
            prompts = prompts[:args.max_samples]
        return prompts, 1


def hook_model(pipeline: DiffusionPipeline, device: tp.Any, args: argparse.Namespace) -> VectorControl:
    if args.command is None:
        return None

    if args.command == 'erase':
        source_concept = unpickle(args.concept_path)
        target_concept = None
    else:
        source_concept = unpickle(args.source_concept_path)
        target_concept = unpickle(args.target_concept_path)

    vector_control = CrossAttentionOutputSteering(
        target_concepts=[target_concept],
        source_concepts=[source_concept],
        strength=args.steering_strength,
        device=device,
        intermediate_clipping=args.intermediate_clipping,
        use_first_diffusion_step=not args.use_all_diffusion_steps,
    )

    model_component = getattr(pipeline, 'transformer', None) or pipeline.unet
    diffusion_register_vector_controls_with_hooks(
        model_component,
        vector_control,
        model_type=DiffusionModelType.from_model(args.model_name),
    )
    return vector_control


def main(args: argparse.Namespace):
    if args.steering_strength is None and args.command is not None:
        raise ValueError(f'--steering_strength (float) must be specified for steering')

    if args.command is None and args.steering_strength is not None:
        raise ValueError(f'--steering_strength is provided but no steering action (erase or translate) specified')

    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)
    device = get_device()

    vector_control = hook_model(pipeline, device, args)

    prompts, num_images_per_prompt = load_prompts(args)
    skipped = generated = 0

    print(f'Generating images for concept {args.generate_concept} with strength {args.steering_strength}')
    for prompt in prompts:
        num_batches = math.ceil(num_images_per_prompt / args.batch_size)
        for batch_id in range(0, num_batches):
            seed = args.seed + batch_id
            num_images = min(args.batch_size, num_images_per_prompt - batch_id * args.batch_size)

            output_paths = [f'{args.output_dir}/{prompt}/{seed}-{idx}.{EXTENSIONS[args.file_format]}' for idx in range(num_images)]
            if all(os.path.exists(path) for path in output_paths):
                skipped += num_images
                continue
            generated += num_images
            images = run_image_model(
                model_type=args.model_name,
                pipe=pipeline,
                prompt=prompt,
                seed=seed,
                device=device,
                num_images=num_images,
            )
            if vector_control is not None:
                vector_control.reset()
            os.makedirs(os.path.dirname(output_paths[0]), exist_ok=True)
            for path, image in zip(output_paths, images):
                image.save(path, format=args.file_format, **SAVE_OPTIONS[args.file_format])

    print(f'Skipped {skipped} images, generated {generated} images')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main_parser = parser.add_argument_group('Common arguments')

    # Generation params
    main_parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True,
                             help='Diffusion model name used for generation')
    main_parser.add_argument('--generate_concept', type=str, required=True, help='Concept for which to generate images')
    main_parser.add_argument('--output_dir', type=str, required=True, help='Directory where generated images should be written')
    main_parser.add_argument('--num_images_per_prompt', type=int, default=10, help='Number of images to generate for each prompt')
    main_parser.add_argument('--batch_size', type=int, default=1, help='Batch size used for image generation')
    main_parser.add_argument('--seed', type=int, default=0, help='Starting seed for each prompt')
    main_parser.add_argument('--file_format', type=str, choices=['PNG', 'JPEG'], default='PNG', help='File format for generated images')
    main_parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use from the dataset')
    main_parser.add_argument('--template_path', type=str, default=None, help='Path to template JSON for evaluation prompts (default: imagenet template)')

    # Steering params
    main_parser.add_argument('--steering_strength', type=float, default=None, help='Steering strength beta (default for erasure: 2.0)')
    main_parser.add_argument('--intermediate_clipping', action='store_true', help='Apply intermediate clipping (Eq. 6)')
    main_parser.add_argument('--use_all_diffusion_steps', action='store_true', help='Use per-step steering vectors instead of single-step')

    subparsers = parser.add_subparsers(dest='command')

    # Params for concept erasure
    erase_parser = subparsers.add_parser('erase')
    erase_parser.add_argument('--concept_path', type=str, required=True,
                              help='Path to concept vectors to erase')

    # Params for concept switching
    translate_parser = subparsers.add_parser('translate')
    translate_parser.add_argument('--source_concept_path', type=str, required=True,
                                  help='Path to source concept vectors')
    translate_parser.add_argument('--target_concept_path', type=str, required=True,
                                  help='Path to target concept vectors')

    args = parser.parse_args()

    main(args)
