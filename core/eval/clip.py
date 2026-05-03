import glob
import operator
from functools import reduce
import torch
import clip
import numpy as np
from typing import List, Union
from PIL import Image
from tqdm import tqdm


from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

EXTENSIONS = ["png", "jpg"]


def get_clip_preprocess(n_px=224):
    def Convert(image):
        return image.convert("RGB")

    image_preprocess = Compose(
        [
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    def text_preprocess(text):
        return clip.tokenize(text, truncate=True)

    return image_preprocess, text_preprocess


@torch.no_grad()
def clip_score(
    images: List[str],
    texts: str,
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
    cross_matching: bool = False,
    batch_size: int = 50,
    device: str | None = None,
):
    """
    Compute CLIPScore (https://arxiv.org/abs/2104.08718) for generated images according to their prompts.
    *Important*: same as the official implementation, we take *SUM* of the similarity scores across all the
        reference texts. If you are evaluating on the Concept Erasing task, it might should be modified to *MEAN*,
        or only one reference text should be given.

    Args:
        images (List[Union[torch.Tensor, np.ndarray, PIL.Image.Image, str]]): A list of generated images.
            Can be a list of torch.Tensor, numpy.ndarray, PIL.Image.Image, or a str of image path.
        texts (str): A list of prompts.
        w (float, optional): The weight of the similarity score. Defaults to 2.5.
        clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
        n_px (int, optional): The size of images. Defaults to 224.
        cross_matching (bool, optional): Whether to compute the similarity between images and texts in cross-matching manner.

    Returns:
        score (np.ndarray): The CLIPScore of generated images.
            size: (len(images), )
    """
    if isinstance(texts, str):
        texts = [texts]

    assert len(images) == len(
        texts
    ), "The length of images and texts should be the same if cross_matching is False."

    if cross_matching:
        raise NotImplementedError("cross_matching=True is not implemented in this helper")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = clip.load(clip_model, device=device)
    image_preprocess, text_preprocess = get_clip_preprocess(
        n_px
    )  # following the official implementation, rather than using the default CLIP preprocess

    sc = []
    for start in tqdm(range(0, len(images), batch_size)):
        end = min(start + batch_size, len(images))
        batch_images = images[start:end]
        batch_texts = texts[start:end]

        # extract all texts
        texts_feats = text_preprocess(batch_texts).to(device)
        texts_feats = model.encode_text(texts_feats)
    
        # extract all images
        images_feats = [Image.open(img) for img in batch_images]
        images_feats = [image_preprocess(img) for img in images_feats]
        images_feats = torch.stack(images_feats, dim=0).to(device)
        images_feats = model.encode_image(images_feats)
    
        # compute the similarity
        images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
        texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)
        
        score = w * images_feats * texts_feats
        sc.append(score)

    if not sc:
        return np.asarray([], dtype=np.float32)

    sc = torch.cat(sc, dim=0)
    return sc.sum(dim=1).clamp(min=0).cpu().numpy()




@torch.no_grad()
def clip_accuracy(
    images: List[str],
    ablated_texts: Union[List[str], str],
    anchor_texts: Union[List[str], str],
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
):
    """
    Compute CLIPAccuracy according to CLIPScore.

    Args:
        images (List[Union[torch.Tensor, np.ndarray, PIL.Image.Image, str]]): A list of generated images.
            Can be a list of torch.Tensor, numpy.ndarray, PIL.Image.Image, or a str of image path.
        ablated_texts (Union[List[str], str]): A list of prompts that are ablated from the anchor texts.
        anchor_texts (Union[List[str], str]): A list of prompts that the ablated concepts fall back to.
        w (float, optional): The weight of the similarity score. Defaults to 2.5.
        clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
        n_px (int, optional): The size of images. Defaults to 224.

    Returns:
        accuracy (float): The CLIPAccuracy of generated images. size: (len(images), )
    """
    if isinstance(ablated_texts, str):
        ablated_texts = [ablated_texts]
    if isinstance(anchor_texts, str):
        anchor_texts = [anchor_texts]

    assert len(ablated_texts) == len(
        anchor_texts
    ), "The length of ablated_texts and anchor_texts should be the same."

    ablated_clip_score = clip_score(images, ablated_texts, w, clip_model, n_px)
    anchor_clip_score = clip_score(images, anchor_texts, w, clip_model, n_px)
    accuracy = np.mean(anchor_clip_score < ablated_clip_score).item()

    return accuracy


def clip_eval_by_image(
    images: List[str],
    concept,
    eval_with_template = False,
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
):
    """
    Compute CLIPScore and CLIPAccuracy with generated images.

    Args:
        images (List[Union[torch.Tensor, np.ndarray, PIL.Image.Image, str]]): A list of generated images.
            Can be a list of torch.Tensor, numpy.ndarray, PIL.Image.Image, or a str of image path.
        ablated_texts (Union[List[str], str]): A list of prompts that are ablated from the anchor texts.
        anchor_texts (Union[List[str], str]): A list of prompts that the ablated concepts fall back to.
        w (float, optional): The weight of the similarity score. Defaults to 2.5.
        clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
        n_px (int, optional): The size of images. Defaults to 224.

    Returns:
        score (float): The CLIPScore of generated images.
        accuracy (float): The CLIPAccuracy of generated images.
    """
    num_images = len(images)
    target_prompts = [concept] * num_images
    anchor_prompts = [""] * num_images
                
    ablated_clip_score = clip_score(images, target_prompts, w, clip_model, n_px)
    anchor_clip_score = clip_score(images, anchor_prompts, w, clip_model, n_px)
    accuracy = np.mean(anchor_clip_score < ablated_clip_score).item()
    score = np.mean(ablated_clip_score).item()

    return score, accuracy



def compute_clip(
    path: str,
    concept: str,
    fname: str = None,
):
    if fname is not None:
        images = glob.glob(f'{path}/**/{fname}', recursive=True)
    else:
        images = reduce(operator.add, [glob.glob(f'{path}/**/*.{ext}', recursive=True) for ext in EXTENSIONS])
    score, accuracy = clip_eval_by_image(images, concept)
    return score, accuracy
