"""
Configuration for Palm Biometric System
========================================
Designed for: 500-600 people, single-shot RGB+IR identification.
Target: 90%+ Top-1 accuracy.

All hyperparameters tuned for this scale.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # =============================================================
    # PATHS
    # =============================================================
    # IMPORTANT: Use the ROI-extracted dataset, NOT raw images.
    # Run palm_roi.py first to create this folder.
    data_dir: str = "final_folder_roi"
    
    # Where the trained model and metadata go
    save_dir: str = "models"
    
    # Where the enrolled person database goes
    database_dir: str = "enrolled_database"
    
    # Logs
    log_dir: str = "logs"
    
    # =============================================================
    # MODEL ARCHITECTURE
    # =============================================================
    backbone: str = "resnet50"          # ResNet50 pretrained on ImageNet
    embedding_dim: int = 512            # Larger embedding for 500+ classes
    dropout: float = 0.4                # Higher dropout to combat small dataset
    pretrained: bool = True
    
    # =============================================================
    # ARCFACE LOSS PARAMETERS
    # =============================================================
    arcface_scale: float = 30.0         # s parameter (logit scale)
    arcface_margin: float = 0.5         # m parameter (angular margin)
    
    # =============================================================
    # JOINT LOSS WEIGHTS
    # =============================================================
    weight_arcface: float = 1.0         # Main metric learning loss
    weight_ce: float = 1.0              # Standard classification
    label_smoothing: float = 0.1        # Reduces overfitting
    
    # =============================================================
    # TRAINING SETTINGS
    # =============================================================
    num_epochs: int = 150
    batch_size: int = 32                # Per epoch all 1000 images cycle multiple times
    
    # With only 2 images per person, we cycle the data with augmentation
    # Effectively each "epoch" means N augmented passes through the data
    augmentation_passes_per_epoch: int = 8
    
    # =============================================================
    # OPTIMIZER
    # =============================================================
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    momentum: float = 0.9               # For SGD if used
    optimizer: str = "adamw"            # adamw or sgd
    
    # =============================================================
    # LEARNING RATE SCHEDULE
    # =============================================================
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Backbone freezing schedule (helps small datasets)
    freeze_backbone_epochs: int = 20    # Train only head for first N epochs
    
    # =============================================================
    # DATA AUGMENTATION
    # =============================================================
    # All augmentations are applied IDENTICALLY to RGB and IR (paired)
    aug_rotation_degrees: float = 15.0
    aug_translate_pct: float = 0.08
    aug_scale_range: Tuple[float, float] = (0.85, 1.15)
    aug_horizontal_flip_prob: float = 0.5
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_random_erasing_prob: float = 0.25
    
    # =============================================================
    # EARLY STOPPING
    # =============================================================
    patience: int = 25
    min_delta: float = 0.001
    
    # =============================================================
    # VALIDATION
    # =============================================================
    # We use Image 1 (rgb_0/ir_0) for training, Image 2 (rgb_1/ir_1) for val.
    # If your data only has one of each per person, this falls back to
    # leave-one-out style with augmentation.
    val_split_strategy: str = "second_image"  # "second_image" or "fraction"
    val_fraction: float = 0.2                  # Used only if strategy = "fraction"
    
    # =============================================================
    # TEST-TIME AUGMENTATION (TTA)
    # =============================================================
    use_tta: bool = True
    num_tta_augmentations: int = 5
    
    # =============================================================
    # CONFIDENCE THRESHOLDING
    # =============================================================
    # Cosine similarity threshold above which a match is accepted.
    # Calibrated automatically during validation.
    default_match_threshold: float = 0.5
    target_far: float = 0.01            # Target False Accept Rate (1%)
    
    # =============================================================
    # CHECKPOINTING
    # =============================================================
    save_every: int = 10
    
    # =============================================================
    # SYSTEM
    # =============================================================
    device: str = "cpu"                 # "cuda" if GPU available
    num_workers: int = 4
    seed: int = 42
    num_cpu_threads: int = 8
    
    # =============================================================
    # IMAGE SETTINGS
    # =============================================================
    image_size: Tuple[int, int] = (224, 224)
    rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ir_mean: float = 0.45
    ir_std: float = 0.25
    
    def __post_init__(self):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.database_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        """Serialize for saving with checkpoint."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d


DEFAULT_CONFIG = Config()