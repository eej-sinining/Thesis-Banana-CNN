import os
from typing import Dict, List, Tuple

import numpy as np  # type: ignore
from PIL import Image, ImageDraw, ImageFont, ImageEnhance  # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

from config import BASE_DIR
from utils.models import baseline_predict, enhanced_predict
from utils.models.common import preprocess_image

CLASS_LABELS = ["Cordana", "Sanas", "SigatokaNegra"]
CLASS_DISPLAY = {
    "Cordana": "Cordana",
    "Sanas": "Healthy",
    "SigatokaNegra": "Black Sigatoka",
}

MODEL_METADATA = {
    "baseline": {
        "name": "Baseline ResNet50",
        "version": "v5",
        "type": "TFLite deployment",
        "notes": "Standard ResNet50 inference pipeline used as benchmark model.",
        "reference": "utils/artifacts/v4_Thesis_Resnet50.ipynb",
    },
    "enhanced": {
        "name": "Enhanced ResNet50",
        "version": "v5",
        "type": "TFLite deployment",
        "notes": "Optimized ResNet50 variant used for improved performance.",
        "reference": "utils/artifacts/v4_Thesis_Resnet50.ipynb",
    },
}


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _metric_bundle_from_confusion_matrix(cm: np.ndarray) -> Dict[str, object]:
    total = int(cm.sum())
    num_classes = len(CLASS_LABELS)

    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    per_class_metrics: List[Dict[str, float]] = []

    for idx in range(num_classes):
        tp = float(cm[idx, idx])
        fn = float(cm[idx, :].sum() - tp)
        fp = float(cm[:, idx].sum() - tp)
        tn = float(total - (tp + fn + fp))

        support = float(cm[idx, :].sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        specificity = _safe_div(tn, tn + fp)

        weight = _safe_div(support, total)
        weighted_precision += precision * weight
        weighted_recall += recall * weight
        weighted_f1 += f1 * weight

        per_class_metrics.append(
            {
                "class_key": CLASS_LABELS[idx],
                "class_name": CLASS_DISPLAY.get(CLASS_LABELS[idx], CLASS_LABELS[idx]),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "specificity": round(specificity, 4),
                "support": int(support),
            }
        )

    accuracy = _safe_div(float(np.trace(cm)), float(total))
    return {
        "accuracy": round(accuracy, 4),
        "precision_weighted": round(weighted_precision, 4),
        "recall_weighted": round(weighted_recall, 4),
        "f1_score_weighted": round(weighted_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_order": [CLASS_DISPLAY.get(c, c) for c in CLASS_LABELS],
    }


def _pil_to_model_input(img: Image.Image) -> np.ndarray:
    arr = np.array(img.resize((224, 224)).convert("RGB"), dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr.astype(np.float32)


def _image_variants(image_path: str) -> List[np.ndarray]:
    with Image.open(image_path).convert("RGB") as src:
        base = src.copy()

    variants = [
        base,
        base.transpose(Image.FLIP_LEFT_RIGHT),
        base.rotate(10, resample=Image.BICUBIC),
        base.rotate(-10, resample=Image.BICUBIC),
        ImageEnhance.Brightness(base).enhance(1.15),
        ImageEnhance.Brightness(base).enhance(0.85),
        ImageEnhance.Contrast(base).enhance(1.2),
        ImageEnhance.Contrast(base).enhance(0.85),
        ImageEnhance.Color(base).enhance(1.15),
    ]
    return [_pil_to_model_input(v) for v in variants]


def _class_to_idx(class_name: str) -> int:
    try:
        return CLASS_LABELS.index(class_name)
    except ValueError:
        return 0


def _predictor_for_model(model_key: str):
    return enhanced_predict if model_key == "enhanced" else baseline_predict


def _build_consistency_metrics(image_path: str, model_key: str) -> Dict[str, object]:
    predictor = _predictor_for_model(model_key)
    samples = _image_variants(image_path)

    predicted_classes: List[str] = []
    for sample in samples:
        result = predictor(sample)
        predicted_classes.append(str(result.get("disease", "Unknown")))

    valid_classes = [c for c in predicted_classes if c in CLASS_LABELS]
    if not valid_classes:
        valid_classes = ["Sanas"]

    unique, counts = np.unique(valid_classes, return_counts=True)
    pseudo_true = str(unique[np.argmax(counts)])
    pseudo_idx = _class_to_idx(pseudo_true)

    cm = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int32)
    for pred in valid_classes:
        cm[pseudo_idx, _class_to_idx(pred)] += 1

    metrics = _metric_bundle_from_confusion_matrix(cm)
    metrics["sample_count"] = len(samples)
    metrics["pseudo_true_class"] = CLASS_DISPLAY.get(pseudo_true, pseudo_true)
    metrics["consistency_score"] = round(
        _safe_div(float(max(counts)), float(len(valid_classes))), 4
    )
    return metrics


# ─────────────────────────────────────────────────────────────
# OCCLUSION SENSITIVITY BBOX  (ported from notebook Cell 20)
# ─────────────────────────────────────────────────────────────

def _occlusion_saliency_map(
    predictor,
    img_array: np.ndarray,
    pred_idx: int,
    patch_size: int = 48,
    stride: int = 24,
) -> np.ndarray:
    """
    img_array: shape (1, 224, 224, 3), already preprocess_input-normalised.
    Returns a (224, 224) float32 saliency map normalised to [0, 1].
    """
    H, W = 224, 224
    base_result = predictor(img_array)
    # predictor returns {"disease": str, "confidence": float, ...}
    # We need raw class probabilities; fall back to confidence of predicted class.
    # To get the drop for *pred_idx* we use the confidence value directly when
    # the predicted class matches, else treat drop as 0.
    base_conf = float(base_result.get("confidence", 0.0)) / 100.0

    saliency = np.zeros((H, W), dtype=np.float32)
    counts   = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = img_array.copy()
            # Neutral fill in preprocess_input space (ImageNet mean → 0)
            occluded[0, y : y + patch_size, x : x + patch_size, :] = 0.0
            occ_result = predictor(occluded)
            occ_cls    = occ_result.get("disease", "")
            occ_conf   = float(occ_result.get("confidence", 0.0)) / 100.0

            # If occluding changed the prediction the drop is base_conf (maximum)
            if occ_cls != base_result.get("disease"):
                drop = base_conf
            else:
                drop = max(0.0, base_conf - occ_conf)

            saliency[y : y + patch_size, x : x + patch_size] += drop
            counts  [y : y + patch_size, x : x + patch_size] += 1.0

    counts   = np.where(counts == 0, 1, counts)
    saliency /= counts
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency


def _saliency_to_bbox(
    saliency: np.ndarray, threshold: float = 0.35
) -> Tuple[int, int, int, int] | None:
    binary = (saliency >= threshold).astype(np.uint8)

    if binary.sum() == 0:
        for t in (0.25, 0.2, 0.15, 0.1):
            binary = (saliency >= t).astype(np.uint8)
            if binary.sum() > 0:
                break

    if binary.sum() == 0:
        return None

    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    pad = 8
    x0 = max(0,   int(x0) - pad)
    y0 = max(0,   int(y0) - pad)
    x1 = min(223, int(x1) + pad)
    y1 = min(223, int(y1) + pad)
    return x0, y0, x1, y1


def _draw_bbox_on_image(
    img_pil: Image.Image,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (220, 50, 50),
    lw: int = 2,
) -> Image.Image:
    out  = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = bbox

    for d in range(lw):
        draw.rectangle([x0 - d, y0 - d, x1 + d, y1 + d], outline=color)

    text = f"{label} {confidence:.1f}%"
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 8
        )
    except Exception:
        font = ImageFont.load_default()

    tb  = draw.textbbox((0, 0), text, font=font)
    tw  = tb[2] - tb[0]
    th  = tb[3] - tb[1]
    pad = 1
    ly  = max(0, y0 - th - 2 * pad - lw)
    draw.rectangle([x0, ly, min(out.width - 1, x0 + tw + 2 * pad), ly + th + 2 * pad], fill=color)
    draw.text((x0 + pad, ly + pad), text, fill=(255, 255, 255), font=font)
    return out


def create_annotation(
    image_path: str,
    label: str,
    confidence: float,
    variant: str,
    model_key: str = "enhanced",
) -> Dict[str, object]:
    """
    Runs occlusion-sensitivity on the uploaded image to find the disease region,
    then draws a tight bounding box around it.
    Falls back to a centre-crop box if the saliency map yields nothing.
    """
    annotations_dir = os.path.join(BASE_DIR, "static", "uploads", "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # ── load & resize to 224 × 224 (model input size) ──
    with Image.open(image_path).convert("RGB") as src:
        img_224 = src.resize((224, 224))

    # ── build preprocess_input array ──
    arr = np.array(img_224, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr).astype(np.float32)

    predictor = _predictor_for_model(model_key)
    pred_result = predictor(arr)
    pred_cls    = pred_result.get("disease", "Sanas")

    # ── occlusion saliency → bbox ──
    saliency = _occlusion_saliency_map(predictor, arr, _class_to_idx(pred_cls))
    bbox     = _saliency_to_bbox(saliency, threshold=0.35)

    # ── fallback: centre box if nothing found ──
    if bbox is None:
        bbox = (45, 45, 179, 179)

    # ── choose colour: green for healthy, red for disease ──
    is_healthy = pred_cls == "Sanas"
    color      = (50, 200, 50) if is_healthy else (220, 50, 50)

    annotated = _draw_bbox_on_image(img_224, bbox, label, confidence, color=color)

    base_name   = os.path.basename(image_path)
    name, _     = os.path.splitext(base_name)
    output_name = f"{name}_{variant}_annotation.jpg"
    output_path = os.path.join(annotations_dir, output_name)
    annotated.save(output_path, format="JPEG", quality=92)

    return {
        "file_name": output_name,
        "relative_path": f"uploads/annotations/{output_name}",
        "bbox": {"x_min": bbox[0], "y_min": bbox[1], "x_max": bbox[2], "y_max": bbox[3]},
    }


def build_model_reports(
    image_path: str,
    baseline_result: Dict[str, object],
    enhanced_result: Dict[str, object],
) -> Dict[str, object]:
    baseline_metrics = _build_consistency_metrics(image_path=image_path, model_key="baseline")
    enhanced_metrics = _build_consistency_metrics(image_path=image_path, model_key="enhanced")

    baseline_annotation = create_annotation(
        image_path=image_path,
        label=str(baseline_result.get("disease_name", baseline_result.get("disease", "Unknown"))),
        confidence=float(baseline_result.get("confidence", 0.0)),
        variant="baseline",
        model_key="baseline",
    )
    enhanced_annotation = create_annotation(
        image_path=image_path,
        label=str(enhanced_result.get("disease_name", enhanced_result.get("disease", "Unknown"))),
        confidence=float(enhanced_result.get("confidence", 0.0)),
        variant="enhanced",
        model_key="enhanced",
    )

    return {
        "baseline": {
            "model": MODEL_METADATA["baseline"],
            "metrics": baseline_metrics,
            "annotation": baseline_annotation,
        },
        "enhanced": {
            "model": MODEL_METADATA["enhanced"],
            "metrics": enhanced_metrics,
            "annotation": enhanced_annotation,
        },
    }