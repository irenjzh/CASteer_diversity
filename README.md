# CASteer-Diversity

CASteer-Diversity is an experimental repository for **training-free diversity control** in text-to-image diffusion models. The method increases the variability of images generated from the same text prompt by adding precomputed steering vectors to internal model activations during inference.

The model weights are not fine-tuned. Steering is applied only at inference time.

## Relation to CASteer

This project is based on the idea of **cross-attention steering** from the CASteer paper:

> Gaintseva et al., *CASteer: Cross-Attention Steering for Controllable Concept Erasure*, arXiv:2503.09630.

Original CASteer repository:

```text
https://github.com/Atmyre/CASteer
```

The original CASteer method computes concept-specific steering vectors from paired positive/negative prompts and applies them to cross-attention outputs for controllable concept erasure or concept editing.

This repository adapts the same idea to a different objective: **diversity control**. Instead of removing a single concept, the pipeline builds a bank of steering directions and applies them as controlled activation shifts in order to increase the diversity of generated images while monitoring quality and text alignment.

## Main differences from the original CASteer setup

- Original CASteer: concept erasure / concept editing.
- This repository: diversity enhancement for text-to-image generation.
- Original CASteer: usually targets a specific concept.
- This repository: uses a bank of style and concrete visual concepts.
- Original CASteer: evaluates concept removal / preservation.
- This repository: evaluates diversity-quality trade-off using CLIPScore, PickScore, ImageReward, MPS/MSS, Vendi Score, and FID-to-baseline.
- This repository includes validation and test pipelines for **SDXL** and **Sana**.

## Supported model aliases

The notebook experiments use the following model aliases:

| Alias | Meaning | Typical use |
|---|---|---|
| `sdxl` | Stable Diffusion XL | main SDXL experiments |
| `sdxl-turbo` | distilled SDXL variant | efficient steering-vector estimation |
| `sdxl_cno` | SDXL configuration for CNO comparison | DDIM/CNO-style comparison protocol |
| `sana15` | Sana 1.5 / 1.6B model | main Sana experiments |
| `sana-sprint` | distilled Sana variant | efficient steering-vector estimation |

## Data

The experiments use prompts from **MS-COCO Captions**, in particular:

```text
annotations/captions_val2017.json
```

In the notebook, COCO data and experiment outputs are stored in Google Drive, for example:

```text
/content/drive/CASteer/coco
/content/drive/CASteer/sdxl
/content/drive/CASteer/sana_1.6
```

You can change these paths depending on your local or Colab setup.

## Installation

The notebook is designed for Google Colab / Linux-like environments.

```bash
git clone https://github.com/irenjzh/CASteer_diversity.git
cd CASteer_diversity

# The notebook resets several packages before installing the repository requirements.
pip uninstall -y transformers tokenizers image-reward sentence-transformers
pip install -r requirements/linux.txt
```

If Hugging Face authentication is required for model access, set the token before running model-loading scripts:

```bash
export HF_TOKEN=your_huggingface_token
```

In Google Colab, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 1. Create COCO validation/test manifests

The notebook creates deterministic COCO-based manifests using `core.create_dataset`.

```python
from core.create_dataset import create_coco_split_manifests, save_json
import os

coco_dir = "./CASteer/coco"
output_dir = "./CASteer/sdxl"

splits = create_coco_split_manifests(
    coco_dir=coco_dir,
    output_dir=output_dir,
    validation_size=20,
    test_size=2000,
    seeds=[0, 1, 2, 3, 4],
)
```

For CNO-style comparison runs, the notebook also creates a separate manifest:

```python
from core.create_dataset import create_coco_split_manifests, save_json
import os

coco_dir = "./CASteer/coco"
splits_dir = "./CASteer/sdxl"
output_dir = os.path.join(splits_dir, "test_cno")

payload = create_coco_split_manifests(
    coco_dir=coco_dir,
    output_dir=output_dir,
    validation_size=0,
    test_size=900,
    split_seed=42,
    seeds=[0, 1],
    overwrite=False,
    create_validation_reference=False,
    create_test_reference=False,
)

save_json(
    os.path.join(output_dir, "test_manifest_cno.json"),
    payload["test_manifest"],
)
```

## 2. Estimate steering-vector banks

### SDXL steering bank

The notebook estimates the SDXL steering bank using the distilled `sdxl-turbo` model and then applies the resulting vectors to the full SDXL pipeline.

```bash
cd /content/CASteer_diversity

DRIVE_ROOT="./CASteer"
STEERING_BANK_DIR="$DRIVE_ROOT/steering_vectors_sdxl"

PYTHONPATH=/content/CASteer_diversity python scripts/diffusion/estimate_steering_vectors.py \
  --model_name sdxl-turbo \
  --run_mode default_bank \
  --num_prompts 50 \
  --output_dir "$STEERING_BANK_DIR"
```

Expected output:

```text
/content/drive/CASteer/steering_vectors_sdxl/default_bank.pickle
```

### Sana steering bank

The notebook estimates the Sana steering bank using `sana-sprint` and applies it to `sana15`.

```bash
cd /content/CASteer_diversity

STEERING_BANK_DIR="$DRIVE_ROOT/steering_vectors_sana_1.6"

PYTHONPATH=/content/CASteer_diversity python scripts/diffusion/estimate_steering_vectors.py \
  --model_name sana-sprint \
  --run_mode default_bank \
  --num_prompts 50 \
  --output_dir "$STEERING_BANK_DIR"
```

Expected output:

```text
/content/drive/CASteer/steering_vectors_sana_1.6/default_bank.pickle
```

## 3. Run validation strength sweep

Validation sweeps are used to select the steering strength `alpha`.

### SDXL validation sweep

```python
from core.create_dataset import create_coco_split_manifests
from core.generate_dataset import run_validation_strength_sweep
import os


splits = create_coco_split_manifests(
    coco_dir=coco_dir,
    output_dir=output_dir,
    validation_size=20,
    test_size=2000,
    seeds=[0, 1, 2, 3, 4],
)

rows, best = run_validation_strength_sweep(
    manifest=splits["validation_manifest"],
    reference_dir=splits["validation_reference_dir"],
    steering_source=steering_source,
    strengths=[0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2, 2.5, 3, 3.5, 4.0, 5.0],
    output_root=output_root,
    model_name="sdxl",
    split="validation",
    file_format="PNG",
    include_baseline=True,
)

print(best)
```

The thesis experiments selected:

```text
SDXL: alpha = 5.0
```

### Sana validation sweep

```python
from core.generate_dataset import run_validation_strength_sweep
import os


rows, best = run_validation_strength_sweep(
    manifest=validation_manifest,
    reference_dir=validation_reference_dir,
    steering_source=steering_source,
    strengths=[0.3, 0.5, 0.8, 1.0, 1.5, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    output_root=output_root,
    model_name="sana15",
    split="validation",
    file_format="PNG",
)

print(best)
```

The thesis experiments selected:

```text
Sana: alpha = 20.0
```

Validation outputs include:

```text
summary.csv
summary.json
best_config.json
clip_vs_strength.png
fid_vs_strength.png
```

## 4. Generate test images

### SDXL test generation

```bash
cd /content/CASteer_diversity

OUTPUT_DIR="./CASteer/sdxl"
OUTPUT_ROOT="$OUTPUT_DIR/test_runs"
STEERING_SOURCE="./CASteer/steering_vectors_sdxl/default_bank.pickle"

python scripts/diffusion/run_manifest_generation.py \
  --model_name sdxl \
  --manifest_path "$OUTPUT_DIR/splits/test_manifest.json" \
  --output_dir "$OUTPUT_ROOT" \
  --variants baseline best_steering random_steering \
  --steering_source "$STEERING_SOURCE" \
  --steering_strength 5.0 \
  --file_format PNG \
  --max_prompts 100
```

### Sana test generation

```bash
cd /content/CASteer_diversity

OUTPUT_DIR="./sana_1.6"
OUTPUT_ROOT="$OUTPUT_DIR/test_runs"
STEERING_SOURCE="./CASteer/steering_vectors_sana_1.6/default_bank.pickle"

python scripts/diffusion/run_manifest_generation.py \
  --model_name sana15 \
  --manifest_path "$OUTPUT_DIR/splits/test_manifest.json" \
  --output_dir "$OUTPUT_ROOT" \
  --variants baseline best_steering random_steering \
  --steering_source "$STEERING_SOURCE" \
  --steering_strength 20.0 \
  --file_format PNG \
  --max_prompts 100
```

## 5. Evaluate test metrics

The notebook evaluates the following metrics:

```text
clip_score, pick_score, image_reward, mps, vendi
```

Run evaluation for SDXL or Sana by pointing `--output_root` to the corresponding test directory:

```bash
python scripts/diffusion/evaluate_test_metrics.py \
  --output_root "$OUTPUT_ROOT" \
  --split test \
  --metrics clip_score pick_score image_reward mps vendi
```

The evaluator writes per-variant metrics and summary files into the experiment directory.

## 6. CNO-style comparison run

The notebook also contains a CNO-comparison configuration based on `sdxl_cno`.

```bash
cd /content/CASteer_diversity

SPLITS_DIR="./CASteer/sdxl"
MANIFEST_PATH="$SPLITS_DIR/test_cno/test_manifest_cno.json"
OUTPUT_DIR="$SPLITS_DIR/test_cno_ddim/test_photos"

python3 scripts/diffusion/run_manifest_generation.py \
  --model_name sdxl_cno \
  --steering_strength 6 \
  --steering_source "$STEERING_SOURCE" \
  --manifest_path "$MANIFEST_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --variants baseline best_steering \
  --file_format PNG \
  --max_prompts 150
```

Then evaluate:

```bash
python scripts/diffusion/evaluate_test_metrics.py \
  --output_root "$OUTPUT_DIR" \
  --split test \
  --metrics clip_score pick_score image_reward mps vendi
```

## 7. Inspect model scheduler / inference setup

The notebook checks the pipeline configuration with:

```python
from core.utils import init_pipeline_for_image_model, get_num_denoising_steps

model_name = "sdxl_cno"
pipe = init_pipeline_for_image_model(model_name)

print("model_name:", model_name)
print("scheduler:", pipe.scheduler.__class__.__name__)
print("num_inference_steps:", get_num_denoising_steps(model_name))
```

## Output structure

Typical output directories:

```text
/content/drive/CASteer/
├── steering_vectors_sdxl/
│   └── default_bank.pickle
├── steering_vectors_sana_1.6/
│   └── default_bank.pickle
├── sdxl/
│   ├── splits/
│   ├── validation_strength_sweep/
│   └── test_runs/
└── sana_1.6/
    ├── validation_strength_sweep/
    └── test_runs/
```

## Notes

- The notebook is Colab-oriented and assumes that COCO data and outputs are stored in Google Drive.
- The steering-vector bank is computed once and reused during generation.
- `best_steering` uses a curated concept direction from the steering bank.
- `random_steering` is used as a control condition to separate meaningful steering from random activation perturbation.
- For the thesis experiments, the main test protocol used 100 prompts and 6 generated images per prompt.
- For the CNO comparison, a larger COCO-based protocol was used.

## Citation

If you use the original CASteer idea, cite:

```bibtex
@article{gaintseva2025casteer,
  title={CASteer: Cross-Attention Steering for Controllable Concept Erasure},
  author={Gaintseva, Tatiana and Oncescu, Andreea-Maria and Ma, Chengcheng and Liu, Ziquan and Benning, Martin and Slabaugh, Gregory and Deng, Jiankang and Elezi, Ismail},
  journal={arXiv preprint arXiv:2503.09630},
  year={2025}
}
```

This repository is an experimental adaptation of CASteer for diversity enhancement and is not the official CASteer implementation.
