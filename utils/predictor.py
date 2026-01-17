import random

def predict_image(image_path):
    diseases = [
        "Black Sigatoka",
        "Yellow Sigatoka",
        "Banana Bacterial Wilt",
        "Healthy Leaf"
    ]

    descriptions = {
        "Black Sigatoka": "A fungal disease characterized by dark streaks on banana leaves.",
        "Yellow Sigatoka": "A fungal disease causing yellow spots and reduced photosynthesis.",
        "Banana Bacterial Wilt": "A bacterial infection leading to wilting and yellowing.",
        "Healthy Leaf": "The banana leaf appears healthy with no visible disease symptoms."
    }

    disease = random.choice(diseases)
    confidence = round(random.uniform(85, 99), 2)

    return {
        "disease": disease,
        "confidence": confidence,
        "description": descriptions[disease]
    }
