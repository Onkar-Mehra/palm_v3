"""
RGB to IR Conversion Module
============================

Your "IR" images are actually grayscale photographs (verified: all 3 channels
are identical, just darker than RGB). This module converts an RGB image to
a matching grayscale "IR" image so you can capture only RGB at enrollment
time and still produce both files the model expects.

How it works:
1. Convert RGB to grayscale (luminance)
2. Apply darkening curve to match training IR brightness distribution
   (training IR has ~50% the brightness of RGB)
3. Save as 3-channel for compatibility

Verified on your sample data:
  Training RGB mean intensity: ~94
  Training IR  mean intensity: ~43  (about half)
  Conversion target: produce grayscale at ~50% brightness

NOTE: This is a fallback for when you only have an RGB camera. If you have
an actual IR camera, ALWAYS prefer real IR captures over conversion.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def rgb_to_ir_style(rgb_image: np.ndarray, target_mean: float = 43.0) -> np.ndarray:
    """
    Convert an RGB image to a grayscale "IR-style" image that matches
    the brightness distribution of the training IR data.
    
    Args:
        rgb_image: BGR image from cv2.imread, shape (H, W, 3)
        target_mean: target mean intensity in [0, 255]
                    (43 matches your training IR distribution)
    
    Returns:
        3-channel grayscale image, shape (H, W, 3), where all 3 channels
        are identical (matching how your training IR is stored).
    """
    if rgb_image is None or rgb_image.size == 0:
        raise ValueError("Input image is empty")
    
    # Convert to single-channel grayscale using standard luminance weights
    if len(rgb_image.shape) == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_image.copy()
    
    # Adjust brightness to match training IR distribution
    # Training IR has mean intensity around 43 (very dark)
    current_mean = gray.mean()
    if current_mean > 1.0:
        scale = target_mean / current_mean
        # Apply scaling
        gray = (gray.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
    
    # Stack to 3 channels (matching how training IR is saved)
    ir_3channel = cv2.merge([gray, gray, gray])
    
    return ir_3channel


def convert_rgb_file_to_ir(rgb_path: str, ir_output_path: str) -> bool:
    """Convert an RGB image file to an IR-style file."""
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        print(f"ERROR: Could not read {rgb_path}")
        return False
    
    ir = rgb_to_ir_style(rgb)
    cv2.imwrite(str(ir_output_path), ir, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def convert_folder(input_folder: str, output_folder: str) -> dict:
    """
    Convert all RGB images in a folder structure to IR-style images.
    
    Input structure:
        input_folder/
            person_001/
                rgb_1.jpg
                rgb_2.jpg
                rgb_3.jpg
            person_002/
                ...
    
    Output structure (created):
        output_folder/
            person_001/
                rgb_1.jpg     (copied)
                ir_1.jpg      (generated from rgb_1)
                rgb_2.jpg     (copied)
                ir_2.jpg      (generated from rgb_2)
                rgb_3.jpg     (copied)
                ir_3.jpg      (generated from rgb_3)
    """
    import shutil
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    person_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    converted = 0
    failed = 0
    
    print(f"Converting RGB → IR for {len(person_dirs)} persons...")
    
    for person_dir in person_dirs:
        out_person = output_path / person_dir.name
        out_person.mkdir(parents=True, exist_ok=True)
        
        # Find all RGB files
        rgb_files = sorted([
            f for f in person_dir.iterdir()
            if f.is_file() and 'rgb' in f.name.lower()
            and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ])
        
        if not rgb_files:
            print(f"  Skipping {person_dir.name}: no RGB files found")
            failed += 1
            continue
        
        for rgb_file in rgb_files:
            # Copy RGB
            out_rgb = out_person / rgb_file.name
            shutil.copy(str(rgb_file), str(out_rgb))
            
            # Generate matching IR filename
            # rgb_1.jpg -> ir_1.jpg, rgb_image_2.png -> ir_image_2.png
            ir_name = rgb_file.name.lower().replace('rgb', 'ir')
            out_ir = out_person / ir_name
            
            convert_rgb_file_to_ir(str(rgb_file), str(out_ir))
        
        converted += 1
    
    print(f"\nDone: {converted} persons converted, {failed} failed")
    return {'converted': converted, 'failed': failed, 'total': len(person_dirs)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert RGB images to IR-style grayscale images"
    )
    parser.add_argument('--input', required=True, help='Input folder with RGB images')
    parser.add_argument('--output', required=True, help='Output folder for RGB+IR pairs')
    args = parser.parse_args()
    
    convert_folder(args.input, args.output)