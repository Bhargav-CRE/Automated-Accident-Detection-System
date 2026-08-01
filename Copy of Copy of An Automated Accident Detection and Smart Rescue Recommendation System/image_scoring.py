"""
Image scoring for scene classification.
Scores images against text prompts for label assignment.
"""

import os
from PIL import Image

# Lazy-loaded state
_model = None
_preprocess = None
_device = None

_LABELS = [
    "a road accident",
    "a car crash",
    "damaged vehicle after accident",
    "normal road traffic",
    "cars driving normally on road",
]


def _get_model():
    global _model, _preprocess, _device
    if _model is None:
        import torch
        import clip as _vlm
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _preprocess = _vlm.load("ViT-B/32", device=_device)
    return _model, _preprocess, _device


def score_image(image_path):
    """
    Score an image and return (label, confidence) where
    label is "Accident" or "Non Accident", confidence in 0-100.
    """
    if not os.path.isfile(image_path):
        return "Non Accident", 0.0

    try:
        import torch
        model, preprocess, device = _get_model()
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text = _tokenize_labels(_LABELS, device)

        with torch.no_grad():
            logits_per_image, _ = model(image_tensor, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        accident_score = float(probs[0][0] + probs[0][1] + probs[0][2])
        normal_score = float(probs[0][3] + probs[0][4])

        if accident_score > normal_score:
            label = "Accident"
            confidence = accident_score * 100
        else:
            label = "Non Accident"
            confidence = normal_score * 100

        return label, round(confidence, 1)
    except Exception:
        return "Non Accident", 0.0


def _tokenize_labels(labels, device):
    import clip as _vlm
    return _vlm.tokenize(labels).to(device)
