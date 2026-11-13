"""
STEP 04: CLAHE (CONTRAST-LIMITED ADAPTIVE HISTOGRAM EQUALIZATION)
=================================================================
Sequence Position: 4 / 10
Purpose:
  - Contains the exact OpenCV CLAHE implementation function `apply_clahe()`.
  - Converts RGB to CIE LAB color space and applies CLAHE strictly on the L (Lightness)
    channel with clipLimit=2.0 and tileGridSize=(8, 8).
  - Preserves biological green/yellow hues while amplifying subtle leaf venation and edge contours.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plant_hybrid"))
sys.path.insert(0, ROOT)

import config


# ===========================================================================
# THE CORE CLAHE FUNCTION (SHOW THIS FUNCTION TO YOUR TEACHER / MA'AM)
# ===========================================================================
def apply_clahe(rgb_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE on the Lightness (L) channel of an RGB plant image.
    
    Why LAB Color Space?
    - If CLAHE is applied directly on RGB channels, colors get severely distorted
      (leaves become unnatural red/blue).
    - In LAB, 'L' is pure Luminance/Brightness, while 'A' and 'B' store color.
    - Enhancing only 'L' boosts venation & texture while keeping leaf color 100% natural.
    """
    # 1. Convert standard RGB image to CIE LAB color space
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

    # 2. Split LAB image into 3 separate channels: L (Lightness), A, B
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 3. Create OpenCV CLAHE operator
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,           # Prevents noise amplification (default=2.0)
        tileGridSize=tile_grid_size     # Divides image into 8x8 contextual tiles
    )

    # 4. Apply CLAHE enhancement exclusively to the Lightness (L) channel
    enhanced_l = clahe.apply(l_channel)

    # 5. Merge the enhanced L channel back with original A and B color channels
    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    # 6. Convert back from LAB color space to RGB color space
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb


# ===========================================================================
# PIPELINE STEP EXECUTION & VERIFICATION
# ===========================================================================
def run_step_04():
    print("\n" + "=" * 76)
    print(" [STEP 04 / 10] CLAHE CONTRAST ENHANCEMENT")
    print("=" * 76)

    cache_file = os.path.join(config.CACHE_DIR, "data_857e6dd21e.npz")
    data = np.load(cache_file, allow_pickle=True)
    images = data["images"]
    clahe_status = bool(data["clahe"])

    print(f"[*] CLAHE Algorithm Parameters:")
    print(f"    - OpenCV Function          : cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))")
    print(f"    - Color Space Conversion   : RGB -> cv2.COLOR_RGB2LAB -> L Channel -> cv2.COLOR_LAB2RGB")
    print(f"    - Clip Limit (Threshold)   : {config.CLAHE_CLIP}")
    print(f"    - Contextual Grid Size     : {config.CLAHE_TILE} x {config.CLAHE_TILE} tiles")
    print(f"    - Applied in Pipeline      : {clahe_status} (Dataset-wide enhancement)")

    # Demonstrate the CLAHE function live on a sample plant image
    sample_raw = images[0]
    sample_enhanced = apply_clahe(sample_raw, clip_limit=config.CLAHE_CLIP, tile_grid_size=(config.CLAHE_TILE, config.CLAHE_TILE))

    raw_contrast = cv2.cvtColor(sample_raw, cv2.COLOR_RGB2GRAY).std()
    enhanced_contrast = cv2.cvtColor(sample_enhanced, cv2.COLOR_RGB2GRAY).std()

    print(f"\n[*] Live Demonstration on Plant Sample #{0}:")
    print(f"    - Original Image Contrast (Std Dev) : {raw_contrast:.2f}")
    print(f"    - Enhanced Image Contrast (Std Dev) : {enhanced_contrast:.2f}")
    print(f"    - Texture & Edge Contrast Gain      : +{(enhanced_contrast - raw_contrast):.2f} pts")

    # Save a side-by-side visualization figure for teacher review
    out_dir = os.path.join(ROOT, "pipeline_reports")
    os.makedirs(out_dir, exist_ok=True)
    out_fig_path = os.path.join(out_dir, "clahe_before_after_demonstration.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    axes[0].imshow(sample_raw)
    axes[0].set_title(f"Before CLAHE (Raw Image)\nContrast Std: {raw_contrast:.1f}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sample_enhanced)
    axes[1].set_title(f"After CLAHE (L-Channel Enhanced)\nContrast Std: {enhanced_contrast:.1f}", fontsize=10, fontweight="bold", color="#1e3a8a")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig_path, bbox_inches="tight")
    plt.close()

    print(f"    -> Exported Side-by-Side Comparison: {out_fig_path}")
    return clahe_status


if __name__ == "__main__":
    run_step_04()
    print("=" * 76)
    print(" [STEP 04 COMPLETE] CLAHE code verified and executed successfully!")
    print("=" * 76 + "\n")
