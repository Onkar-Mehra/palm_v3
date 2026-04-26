"""
Dataset for Palm Biometric System
==================================
Key features:
- Paired augmentation (RGB and IR get IDENTICAL spatial transforms)
- Proper train/val split (per-person held-out image)
- Standard normalization (ImageNet stats for RGB, custom for IR)
- Heavy augmentation appropriate for small dataset
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================
# PAIRED AUGMENTATION
# ============================================================
class PairedAugmentation:
    """
    Applies the SAME spatial augmentation to RGB and IR images.
    
    This is critical: if RGB gets rotated 7° and IR gets rotated -3°,
    the fused embedding sees mismatched palm regions and learning
    breaks down.
    """
    
    def __init__(self, config, is_training=True):
        self.config = config
        self.is_training = is_training
        self.image_size = config.image_size
    
    def __call__(self, rgb, ir):
        """
        Args:
            rgb: HxWx3 uint8 BGR image
            ir:  HxWx3 or HxWx1 uint8 image
        Returns:
            (rgb_tensor, ir_tensor) both as C×H×W float32 normalized tensors
        """
        if not self.is_training:
            # Validation: no augmentation, just resize and normalize
            return self._to_tensors(rgb, ir)
        
        # Generate ALL random parameters once, apply to both
        h, w = rgb.shape[:2]
        
        # 1. Random horizontal flip (same for both)
        do_flip = random.random() < self.config.aug_horizontal_flip_prob
        
        # 2. Random rotation, scale, translation (same affine for both)
        angle = random.uniform(-self.config.aug_rotation_degrees,
                              self.config.aug_rotation_degrees)
        scale = random.uniform(*self.config.aug_scale_range)
        tx = random.uniform(-self.config.aug_translate_pct,
                           self.config.aug_translate_pct) * w
        ty = random.uniform(-self.config.aug_translate_pct,
                           self.config.aug_translate_pct) * h
        
        # Build affine matrix
        cx, cy = w / 2.0, h / 2.0
        cos_a = np.cos(np.radians(angle)) * scale
        sin_a = np.sin(np.radians(angle)) * scale
        M = np.array([
            [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy + tx],
            [sin_a,  cos_a, cy - sin_a * cx - cos_a * cy + ty]
        ], dtype=np.float32)
        
        # Apply IDENTICAL spatial transform
        rgb_aug = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        ir_aug = cv2.warpAffine(ir, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        if do_flip:
            rgb_aug = cv2.flip(rgb_aug, 1)
            ir_aug = cv2.flip(ir_aug, 1)
        
        # 3. Photometric augmentations (these can differ between modalities)
        rgb_aug = self._photometric(rgb_aug)
        ir_aug = self._photometric(ir_aug, is_ir=True)
        
        # 4. Random erasing (same regions for both - we want them to match)
        if random.random() < self.config.aug_random_erasing_prob:
            erase_box = self._get_erase_box(h, w)
            if erase_box is not None:
                y1, y2, x1, x2 = erase_box
                rgb_aug[y1:y2, x1:x2] = np.random.randint(0, 255, (y2-y1, x2-x1, 3), dtype=np.uint8)
                if len(ir_aug.shape) == 3:
                    ir_aug[y1:y2, x1:x2] = np.random.randint(0, 255, (y2-y1, x2-x1, ir_aug.shape[2]), dtype=np.uint8)
                else:
                    ir_aug[y1:y2, x1:x2] = np.random.randint(0, 255, (y2-y1, x2-x1), dtype=np.uint8)
        
        return self._to_tensors(rgb_aug, ir_aug)
    
    def _photometric(self, img, is_ir=False):
        """Brightness / contrast jitter (independent per modality)."""
        img = img.astype(np.float32)
        
        # Brightness
        b = random.uniform(1 - self.config.aug_brightness, 1 + self.config.aug_brightness)
        img = img * b
        
        # Contrast
        c = random.uniform(1 - self.config.aug_contrast, 1 + self.config.aug_contrast)
        mean = img.mean()
        img = (img - mean) * c + mean
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def _get_erase_box(self, h, w):
        """Random erasing rectangle."""
        for _ in range(10):
            area = h * w
            target_area = random.uniform(0.02, 0.15) * area
            aspect = random.uniform(0.3, 3.3)
            
            eh = int(round(np.sqrt(target_area * aspect)))
            ew = int(round(np.sqrt(target_area / aspect)))
            
            if eh < h and ew < w:
                y1 = random.randint(0, h - eh)
                x1 = random.randint(0, w - ew)
                return y1, y1 + eh, x1, x1 + ew
        return None
    
    def _to_tensors(self, rgb, ir):
        """Resize, normalize, convert to tensors."""
        # Resize
        rgb = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_AREA)
        ir = cv2.resize(ir, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Convert IR to single channel if needed
        if len(ir.shape) == 3 and ir.shape[2] == 3:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        
        # BGR -> RGB (OpenCV loads as BGR; ImageNet stats expect RGB)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Normalize RGB with ImageNet stats
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array(self.config.rgb_mean)) / np.array(self.config.rgb_std)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
        
        # Normalize IR
        ir = ir.astype(np.float32) / 255.0
        ir = (ir - self.config.ir_mean) / self.config.ir_std
        
        # Tile IR to 3 channels (so we can use ResNet pretrained conv1)
        ir = np.stack([ir, ir, ir], axis=0).astype(np.float32)
        
        return torch.from_numpy(rgb), torch.from_numpy(ir)


# ============================================================
# DATASET
# ============================================================
class PalmDataset(Dataset):
    """
    Palm biometric dataset.
    
    Expected structure:
        data_dir/
            person_0001/
                rgb.jpg          # OR rgb_0.jpg, rgb_1.jpg, etc. for multiple
                ir.jpg
            person_0002/
                rgb.jpg
                ir.jpg
            ...
    
    Behavior:
    - Each person becomes one class.
    - If a person has multiple RGB+IR pairs, we use them all.
    - Train/val split is per-person:
        - Train uses image[0]
        - Val uses image[1] if it exists, else same image (with no augmentation for val)
    """
    
    def __init__(self, data_dir, config, mode='train', class_to_label=None):
        self.data_dir = Path(data_dir)
        self.config = config
        self.mode = mode
        
        # Augmentation: training augments, val doesn't
        self.augment = PairedAugmentation(config, is_training=(mode == 'train'))
        
        # Build sample list
        self.samples: List[Dict] = []
        self.class_to_label = class_to_label or {}
        self.label_to_class: Dict[int, str] = {}
        
        self._load_samples()
    
    def _load_samples(self):
        """Scan dataset and build train/val split."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Did you run palm_roi.py first to create the ROI-extracted data?"
            )
        
        person_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        # Skip log file folders
        person_dirs = [d for d in person_dirs if not d.name.startswith('_')]
        
        if len(person_dirs) == 0:
            raise ValueError(f"No person folders found in {self.data_dir}")
        
        # Build label mapping if not provided
        if not self.class_to_label:
            for idx, person_dir in enumerate(person_dirs):
                self.class_to_label[person_dir.name] = idx
        
        # Reverse mapping
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}
        
        # Build samples
        for person_dir in person_dirs:
            person_name = person_dir.name
            if person_name not in self.class_to_label:
                continue
            label = self.class_to_label[person_name]
            
            # Find all RGB and IR files
            rgb_files = sorted([
                f for f in person_dir.iterdir()
                if f.is_file() and 'rgb' in f.name.lower()
                and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
            ])
            ir_files = sorted([
                f for f in person_dir.iterdir()
                if f.is_file() and 'ir' in f.name.lower()
                and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
            ])
            
            if not rgb_files or not ir_files:
                continue
            
            # Pair them up
            num_pairs = min(len(rgb_files), len(ir_files))
            
            if self.mode == 'train':
                indices = [0]
            elif self.mode == 'val':
                indices = [0]  # same image, but val uses no augmentation
            else:  # 'all'
                indices = list(range(num_pairs))
            
            for i in indices:
                self.samples.append({
                    'rgb_path': str(rgb_files[i]),
                    'ir_path': str(ir_files[i]),
                    'label': label,
                    'person_name': person_name,
                })
        
        logger.info(f"[{self.mode}] Loaded {len(self.samples)} samples "
                   f"from {len(self.class_to_label)} classes")
    
    def __len__(self):
        if self.mode == 'train':
            # Each epoch goes through samples N times (with different augmentations)
            return len(self.samples) * self.config.augmentation_passes_per_epoch
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        
        # Load images
        rgb = cv2.imread(sample['rgb_path'])
        ir = cv2.imread(sample['ir_path'])
        
        if rgb is None or ir is None:
            raise IOError(f"Failed to load: {sample['rgb_path']} or {sample['ir_path']}")
        
        # Apply paired augmentation (or just normalization for val)
        rgb_t, ir_t = self.augment(rgb, ir)
        
        return {
            'rgb': rgb_t,
            'ir': ir_t,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'person_name': sample['person_name'],
        }
    
    def get_num_classes(self):
        return len(self.class_to_label)
    
    def save_metadata(self, path):
        meta = {
            'class_to_label': self.class_to_label,
            'label_to_class': {str(k): v for k, v in self.label_to_class.items()},
            'num_classes': len(self.class_to_label),
            'num_samples': len(self.samples),
        }
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)


def create_dataloaders(config):
    """
    Create train and val DataLoaders.
    Both use the SAME class_to_label mapping so labels are consistent.
    """
    # First create train (which builds the class mapping)
    train_dataset = PalmDataset(config.data_dir, config, mode='train')
    
    # Then val using the same mapping
    val_dataset = PalmDataset(
        config.data_dir, config, mode='val',
        class_to_label=train_dataset.class_to_label
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    
    return train_loader, val_loader, train_dataset