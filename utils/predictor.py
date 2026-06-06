import os
import time
from utils.models.common import preprocess_image
from utils.models import baseline_predict, enhanced_predict


DISEASE_INFO = {
    "Cordana": {
        "full_name": "Cordana Leaf Spot",
        "severity": "Moderate",
        "description": "A fungal disease causing oval-shaped spots with gray centers and dark borders.",
        "symptoms": "Small, circular to oval brown spots with yellow halos",
        "treatment": "Remove infected leaves, apply fungicide, improve air circulation"
    },
    "Sanas": {
        "full_name": "Healthy Leaf",
        "severity": "None",
        "description": "The leaf appears healthy with uniform green coloration.",
        "symptoms": "No visible symptoms",
        "treatment": "No treatment needed"
    },
    "SigatokaNegra": {
        "full_name": "Black Sigatoka",
        "severity": "Severe",
        "description": "A destructive banana disease causing dark streaks and rapid leaf death.",
        "symptoms": "Yellow streaks that turn brown or black",
        "treatment": "Remove infected leaves and apply fungicides immediately"
    }
}

def enrich(result):
    info = DISEASE_INFO.get(result["disease"], {})
    return {
        **result,
        "success": True,
        "disease_name": info.get("full_name", result["disease"]),
        "severity": info.get("severity", "Unknown"),
        "description": info.get("description", "No description available"),
        "symptoms": info.get("symptoms", "No symptoms available"),
        "treatment": info.get("treatment", "No treatment available"),
    }

from PIL import Image
import numpy as np
import matplotlib.colors as mcolors

def is_valid_leaf(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # 1. Unique color distribution (filters digital drawings / flat colors)
        small_img = img.resize((100, 100))
        small_np = np.array(small_img)
        unique_colors = len(np.unique(small_np.reshape(-1, 3), axis=0))
        if unique_colors < 1000:
            return False, "Image appears to be a digital drawing, painting, or has artificially flat colors."
            
        # 2. Color Variance (filters solid backgrounds)
        variance = np.var(img_np)
        if variance < 500:
            return False, "Image lacks natural texture and appears to be a solid background or screenshot."
            
        # 3. Green-pixel ratio and hue diversity
        hsv_img = mcolors.rgb_to_hsv(img_np / 255.0)
        h = hsv_img[:,:,0]
        s = hsv_img[:,:,1]
        v = hsv_img[:,:,2]
        
        # Leaf colors (brown/yellow/green) -> hue roughly 0.03 to 0.48
        leaf_mask = ((h >= 0.03) & (h <= 0.48)) & (s > 0.15) & (v > 0.15)
        leaf_ratio = np.sum(leaf_mask) / (hsv_img.shape[0] * hsv_img.shape[1])
        
        if leaf_ratio < 0.15:
            return False, "Image does not contain enough natural leaf colors (green, yellow, brown) or is mostly background."
            
        return True, "Valid"
    except Exception as e:
        return False, f"Validation failed: {str(e)}"

def predict_image(image_path):
    if not os.path.exists(image_path):
        return {"success": False, "error": "Image not found"}

    is_valid, reason = is_valid_leaf(image_path)
    if not is_valid:
        return {
            "success": True,
            "not_a_leaf": True,
            "rejection_reason": reason,
            "baseline": None,
            "enhanced": None
        }

    img_array = preprocess_image(image_path)

    # Baseline prediction
    baseline_result = baseline_predict(img_array)
    # ensure lowercase key
    baseline_result["inference_time_ms"] = baseline_result.get("inference_time_ms") or baseline_result.get("Inference_Time_ms") or 0
    baseline_result = enrich(baseline_result)

    # Enhanced prediction
    enhanced_result = enhanced_predict(img_array)
    enhanced_result["inference_time_ms"] = enhanced_result.get("inference_time_ms") or enhanced_result.get("Inference_Time_ms") or 0
    enhanced_result = enrich(enhanced_result)

    return {
        "success": True,
        "not_a_leaf": False,
        "rejection_reason": None,
        "baseline": baseline_result,
        "enhanced": enhanced_result
    }



