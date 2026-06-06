import os
import json
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from .common import confidence_level
import time

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(
    ARTIFACTS_DIR, "baseline_resnet50_deployment_v9.tflite"
)
CLASS_MAPPING_PATH = os.path.join(
    ARTIFACTS_DIR, "class_indices_v9.json"
)

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class mapping
with open(CLASS_MAPPING_PATH) as f:
    CLASS_MAPPING = json.load(f)

INDEX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

# ONE-TIME WARMUP (runs only once when module loads)
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], dummy)
interpreter.invoke()


def predict(img_array):
    
    # Include preprocessing and TFLite invoke in timing for realistic numbers
    start = time.perf_counter()
    # Make sure img_array is batched (1, 224, 224, 3)
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    inference_time = (time.perf_counter() - start) * 1000  # ms

    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx]) * 100

    return {
        "disease": INDEX_TO_CLASS.get(idx, "Unknown"),
        "confidence": round(conf, 2),
        "confidence_level": confidence_level(conf),
        "all_probabilities": {
            INDEX_TO_CLASS.get(i, str(i)): float(p) * 100
            for i, p in enumerate(preds)
        },
        "inference_time_ms": float(f"{inference_time:.2f}")
    }

