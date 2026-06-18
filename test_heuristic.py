from PIL import Image
import numpy as np
import matplotlib.colors as mcolors

def is_valid_leaf(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # 1. Unique color distribution (filters digital drawings / flat colors)
        # Resize to 100x100 to make unique color counting fast
        small_img = img.resize((100, 100))
        small_np = np.array(small_img)
        # Unique colors in a 100x100 real photo should be well over 1000
        unique_colors = len(np.unique(small_np.reshape(-1, 3), axis=0))
        
        if unique_colors < 1000:
            return False, f"Image appears to be a digital drawing or has artificially flat colors (unique colors: {unique_colors})."
            
        # 2. Color Variance (filters solid backgrounds)
        # Real photos have natural lighting variance
        variance = np.var(img_np)
        if variance < 500:
            return False, "Image lacks natural texture (variance too low)."
            
        # 3. Green-pixel ratio and hue diversity
        hsv_img = mcolors.rgb_to_hsv(img_np / 255.0)
        h = hsv_img[:,:,0]
        s = hsv_img[:,:,1]
        v = hsv_img[:,:,2]
        
        # Leaf colors (brown/yellow/green) -> hue roughly 0.05 to 0.45
        # Require some minimum saturation and brightness to avoid counting gray/black/white
        leaf_mask = ((h >= 0.03) & (h <= 0.48)) & (s > 0.15) & (v > 0.15)
        leaf_ratio = np.sum(leaf_mask) / (hsv_img.shape[0] * hsv_img.shape[1])
        
        if leaf_ratio < 0.15:
            return False, f"Image does not contain enough natural leaf colors (ratio: {leaf_ratio:.2f})."
            
        return True, "Valid"
    except Exception as e:
        return False, str(e)
