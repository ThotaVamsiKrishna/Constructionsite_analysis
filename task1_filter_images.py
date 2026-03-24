#!/usr/bin/env python3
"""
Task 1: Video Quality Assessment

Remove blurry, dark, and duplicate frames from 360° photos.
Data: 310 + 480 = 790 images total
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import json

# Quality check functions

def is_blurry(img_path, threshold=100):
    """Check if image is blurry using Laplacian variance."""
    img = cv2.imread(str(img_path), 0)  # grayscale
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < threshold


def is_dark(img_path, threshold=50):
    """Check if image is too dark using HSV brightness."""
    img = cv2.imread(str(img_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:,:,2])
    return brightness < threshold


def is_duplicate(img1_path, img2_path, threshold=0.95):
    """Check if two images are very similar using SSIM."""
    img1 = cv2.imread(str(img1_path), 0)
    img2 = cv2.imread(str(img2_path), 0)
    # Resize for speed
    img1 = cv2.resize(img1, (1024, 512))
    img2 = cv2.resize(img2, (1024, 512))
    similarity = ssim(img1, img2)
    return similarity > threshold


def filter_folder(input_folder, output_folder):
    """Filter images in a folder and save good ones to output."""
    output_folder.mkdir(parents=True, exist_ok=True)
    images = sorted(list(input_folder.glob("*.JPG")))
    
    kept = 0
    removed_blur = 0
    removed_dark = 0
    removed_dup = 0
    prev_img = None
    
    print(f"Processing {len(images)} images...")
    
    for i, img_path in enumerate(images):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(images)}")
        
        # Check blur
        if is_blurry(img_path):
            removed_blur += 1
            continue
        
        # Check darkness
        if is_dark(img_path):
            removed_dark += 1
            continue
        
        # Check duplicate
        if prev_img and is_duplicate(prev_img, img_path):
            removed_dup += 1
            continue
        
        # Keep it
        kept += 1
        img = cv2.imread(str(img_path))
        cv2.imwrite(str(output_folder / img_path.name), img)
        prev_img = img_path
    
    return {
        'total': len(images),
        'kept': kept,
        'blur': removed_blur,
        'dark': removed_dark,
        'duplicate': removed_dup
    }


if __name__ == "__main__":
    # Folder paths
    base = Path("/home/vamsikrishna/Documents/vamsi_documents/learning/3d_estimation/Task1")
    folder1 = base / "RLT1746244567461/images"
    folder2 = base / "RLT1752866201591/images"
    output_base = base / "filtered_output"
    
    # Process both folders
    print("\nProcessing folder 1...")
    print("="*60)
    stats1 = filter_folder(folder1, output_base / "RLT1746244567461")
    
    print("\nProcessing folder 2...")
    print("="*60)
    stats2 = filter_folder(folder2, output_base / "RLT1752866201591")
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Folder 1: Kept {stats1['kept']}/{stats1['total']} ({stats1['kept']/stats1['total']*100:.0f}%)")
    print(f"  Removed: {stats1['blur']} blur, {stats1['dark']} dark, {stats1['duplicate']} duplicate")
    print(f"\nFolder 2: Kept {stats2['kept']}/{stats2['total']} ({stats2['kept']/stats2['total']*100:.0f}%)")
    print(f"  Removed: {stats2['blur']} blur, {stats2['dark']} dark, {stats2['duplicate']} duplicate")
    
    total = stats1['total'] + stats2['total']
    kept = stats1['kept'] + stats2['kept']
    print(f"\nTotal: Kept {kept}/{total} images ({kept/total*100:.0f}%)")
    print("="*60)
    
    # Save report
    report = {
        'folder1': stats1,
        'folder2': stats2,
        'total_kept': kept,
        'total_processed': total
    }
    
    with open(output_base / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDone! Check {output_base} for results.")
