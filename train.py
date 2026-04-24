"""
Training Script for Palm Biometric System
==========================================
- Joint ArcFace + Cross-Entropy training
- Backbone freezing for first N epochs (helps small datasets)
- Cosine LR schedule with warmup
- Proper validation with held-out images
- Threshold calibration on validation set
- Checkpointing of best model
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path
import numpy as np

# Set CPU threads BEFORE importing torch
NUM_CPU_THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(NUM_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPU_THREADS)

import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(NUM_CPU_THREADS)

from config import Config
from dataset import create_dataloaders
from model import PalmBiometricModel, JointLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_optimizer_and_scheduler(model, config, steps_per_epoch):
    """Build optimizer and LR scheduler."""
    # Different LR for backbone vs head (smaller for pretrained)
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(p)
        else:
            head_params.append(p)
    
    param_groups = [
        {'params': backbone_params, 'lr': config.learning_rate * 0.1},
        {'params': head_params, 'lr': config.learning_rate},
    ]
    
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(param_groups, momentum=config.momentum,
                             weight_decay=config.weight_decay)
    
    # Cosine schedule with warmup
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(config.min_lr / config.learning_rate,
                  0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def _build_post_unfreeze_optimizer(model, config, remaining_epochs, steps_per_epoch):
    """
    Build a fresh optimizer + scheduler when backbones get unfrozen mid-training.
    Uses a short warmup (3 epochs) then cosine decay over the remaining epochs.
    Backbone gets 0.1x LR (fine-tuning), head keeps full LR.
    """
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(p)
        else:
            head_params.append(p)
    
    # Lower base LR for fine-tuning (0.3x of original)
    ft_lr = config.learning_rate * 0.3
    
    param_groups = [
        {'params': backbone_params, 'lr': ft_lr * 0.1},
        {'params': head_params, 'lr': ft_lr},
    ]
    
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(param_groups, momentum=config.momentum,
                             weight_decay=config.weight_decay)
    
    total_steps = max(1, remaining_epochs * steps_per_epoch)
    warmup_steps = max(1, 3 * steps_per_epoch)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(config.min_lr / ft_lr,
                  0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, logger):
    model.train()
    
    total_loss = 0.0
    total_af_acc = 0.0
    total_ce_acc = 0.0
    num_batches = 0
    
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(loader):
        rgb = batch['rgb'].to(device)
        ir = batch['ir'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb, ir, labels)
        loss, metrics = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += metrics['total_loss']
        total_af_acc += metrics['arcface_acc']
        total_ce_acc += metrics['ce_acc']
        num_batches += 1
        
        # Log every 20 batches
        if (batch_idx + 1) % 20 == 0:
            current_lr = optimizer.param_groups[-1]['lr']
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {total_loss/num_batches:.4f} | "
                f"ArcAcc: {total_af_acc/num_batches:.1f}% | "
                f"CEAcc: {total_ce_acc/num_batches:.1f}% | "
                f"LR: {current_lr:.6f}"
            )
    
    epoch_time = time.time() - epoch_start
    return {
        'loss': total_loss / max(1, num_batches),
        'arcface_acc': total_af_acc / max(1, num_batches),
        'ce_acc': total_ce_acc / max(1, num_batches),
        'time_minutes': epoch_time / 60,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, logger):
    """Standard validation - measures loss and accuracy on held-out images."""
    model.eval()
    
    total_loss = 0.0
    total_af_acc = 0.0
    total_ce_acc = 0.0
    num_batches = 0
    
    for batch in loader:
        rgb = batch['rgb'].to(device)
        ir = batch['ir'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(rgb, ir, labels)
        loss, metrics = criterion(outputs, labels)
        
        total_loss += metrics['total_loss']
        total_af_acc += metrics['arcface_acc']
        total_ce_acc += metrics['ce_acc']
        num_batches += 1
    
    return {
        'loss': total_loss / max(1, num_batches),
        'arcface_acc': total_af_acc / max(1, num_batches),
        'ce_acc': total_ce_acc / max(1, num_batches),
    }


@torch.no_grad()
def evaluate_identification(model, train_loader, val_loader, device, logger):
    """
    Real evaluation: build a gallery from training images, then identify
    each validation image by finding the closest gallery embedding.
    
    This simulates real deployment:
    - Gallery = enrolled images (training set)
    - Probe = new captures of the same person (validation set)
    
    Returns Top-1, Top-5, Top-10 accuracies.
    """
    model.eval()
    
    logger.info("Building gallery from training images...")
    
    # Compute mean embedding per class from training set
    class_embeddings = {}
    class_counts = {}
    
    # Use the underlying dataset (not the augmented multi-pass version)
    train_dataset = train_loader.dataset
    base_samples = train_dataset.samples
    
    # For evaluation, we want CLEAN embeddings (no augmentation)
    # Temporarily switch to validation mode augmenter
    orig_augment = train_dataset.augment
    from dataset import PairedAugmentation
    train_dataset.augment = PairedAugmentation(train_dataset.config, is_training=False)
    
    for i, sample in enumerate(base_samples):
        item = train_dataset[i]
        rgb = item['rgb'].unsqueeze(0).to(device)
        ir = item['ir'].unsqueeze(0).to(device)
        emb = model.get_embedding(rgb, ir).cpu().squeeze(0)
        
        label = sample['label']
        if label not in class_embeddings:
            class_embeddings[label] = emb.clone()
            class_counts[label] = 1
        else:
            class_embeddings[label] += emb
            class_counts[label] += 1
    
    # Restore augmenter
    train_dataset.augment = orig_augment
    
    # Compute mean and L2-normalize each prototype
    sorted_labels = sorted(class_embeddings.keys())
    gallery = []
    for lbl in sorted_labels:
        proto = class_embeddings[lbl] / class_counts[lbl]
        proto = proto / (proto.norm() + 1e-12)
        gallery.append(proto)
    gallery = torch.stack(gallery)  # (num_classes, embedding_dim)
    
    logger.info(f"Gallery built: {gallery.shape[0]} classes")
    
    # Now query with validation images
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    total = 0
    
    val_dataset = val_loader.dataset
    
    # Track all genuine and impostor scores for threshold calibration
    genuine_scores = []
    impostor_scores = []
    
    for i, sample in enumerate(val_dataset.samples):
        item = val_dataset[i]
        rgb = item['rgb'].unsqueeze(0).to(device)
        ir = item['ir'].unsqueeze(0).to(device)
        true_label = sample['label']
        
        query_emb = model.get_embedding(rgb, ir).cpu().squeeze(0)
        
        # Cosine similarity with all gallery prototypes
        sims = torch.mv(gallery, query_emb)  # (num_classes,)
        
        # Find true label position in sorted_labels
        try:
            true_idx = sorted_labels.index(true_label)
        except ValueError:
            continue
        
        # Top-K
        top_k = min(10, len(sorted_labels))
        top_scores, top_indices = torch.topk(sims, top_k)
        top_indices = top_indices.tolist()
        
        if top_indices[0] == true_idx:
            top1_hits += 1
        if true_idx in top_indices[:5]:
            top5_hits += 1
        if true_idx in top_indices[:10]:
            top10_hits += 1
        total += 1
        
        # Collect scores for threshold calibration
        genuine_scores.append(sims[true_idx].item())
        # Take a few random impostor scores
        impostor_indices = [j for j in range(len(sorted_labels)) if j != true_idx]
        sampled = random.sample(impostor_indices, min(20, len(impostor_indices)))
        for j in sampled:
            impostor_scores.append(sims[j].item())
    
    if total == 0:
        return None
    
    results = {
        'top1': 100 * top1_hits / total,
        'top5': 100 * top5_hits / total,
        'top10': 100 * top10_hits / total,
        'total': total,
    }
    
    # Calibrate threshold for FAR target
    if genuine_scores and impostor_scores:
        impostor_scores_sorted = sorted(impostor_scores, reverse=True)
        # Threshold above which only top X% of impostors fall (= FAR)
        idx = int(len(impostor_scores_sorted) * 0.01)  # 1% FAR
        threshold_far001 = impostor_scores_sorted[idx] if idx < len(impostor_scores_sorted) else 1.0
        
        # FRR at that threshold (% of genuine below threshold)
        below = sum(1 for s in genuine_scores if s < threshold_far001)
        frr = 100 * below / len(genuine_scores)
        
        results['threshold_at_1pct_far'] = float(threshold_far001)
        results['frr_at_1pct_far'] = float(frr)
    
    return results


def main():
    # Reproducibility
    config = Config()
    set_seed(config.seed)
    
    # Logging
    logger = setup_logging(config.log_dir)
    
    logger.info("=" * 70)
    logger.info("PALM BIOMETRIC TRAINING")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Data dir: {config.data_dir}")
    logger.info("")
    
    # Save config
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Data
    logger.info("Loading data...")
    train_loader, val_loader, train_dataset = create_dataloaders(config)
    num_classes = train_dataset.get_num_classes()
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train samples per epoch: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Train batches per epoch: {len(train_loader)}")
    
    # Save dataset metadata
    train_dataset.save_metadata(save_path / "dataset_metadata.json")
    
    # Model
    logger.info("\nBuilding model...")
    model = PalmBiometricModel(config, num_classes)
    model = model.to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    
    # Loss
    criterion = JointLoss(config)
    
    # Initially freeze backbones
    if config.freeze_backbone_epochs > 0:
        logger.info(f"Freezing backbones for first {config.freeze_backbone_epochs} epochs")
        model.freeze_backbones()
    
    # Optimizer (rebuilt after unfreezing)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config, len(train_loader))
    
    # Training loop
    best_top1 = 0.0
    best_epoch = 0
    history = {'train': [], 'val': [], 'eval': []}
    no_improve_count = 0
    
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    
    for epoch in range(1, config.num_epochs + 1):
        # Unfreeze backbones at the right epoch
        if epoch == config.freeze_backbone_epochs + 1:
            logger.info(f"\n>>> Unfreezing backbones at epoch {epoch} <<<")
            model.unfreeze_backbones()
            # Rebuild optimizer to include the now-trainable backbone params.
            # Build a NEW scheduler that runs cosine over the REMAINING epochs,
            # so backbone fine-tuning gets a clean short warmup then decay.
            remaining_epochs = config.num_epochs - epoch + 1
            steps_per_epoch = len(train_loader)
            optimizer, scheduler = _build_post_unfreeze_optimizer(
                model, config, remaining_epochs, steps_per_epoch
            )
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            config.device, epoch, logger
        )
        
        # Standard validation
        val_metrics = validate(model, val_loader, criterion, config.device, logger)
        
        # Real identification evaluation (every 5 epochs to save time)
        eval_metrics = None
        if epoch % 5 == 0 or epoch == config.num_epochs or epoch == 1:
            logger.info("Running identification evaluation...")
            eval_metrics = evaluate_identification(
                model, train_loader, val_loader, config.device, logger
            )
        
        # Save history
        history['train'].append({'epoch': epoch, **train_metrics})
        history['val'].append({'epoch': epoch, **val_metrics})
        if eval_metrics:
            history['eval'].append({'epoch': epoch, **eval_metrics})
        
        # Log summary
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"EPOCH {epoch}/{config.num_epochs} SUMMARY")
        logger.info("-" * 70)
        logger.info(f"  Train: loss={train_metrics['loss']:.4f}  "
                   f"af_acc={train_metrics['arcface_acc']:.1f}%  "
                   f"ce_acc={train_metrics['ce_acc']:.1f}%")
        logger.info(f"  Val:   loss={val_metrics['loss']:.4f}  "
                   f"af_acc={val_metrics['arcface_acc']:.1f}%  "
                   f"ce_acc={val_metrics['ce_acc']:.1f}%")
        if eval_metrics:
            logger.info(f"  Eval:  Top-1={eval_metrics['top1']:.1f}%  "
                       f"Top-5={eval_metrics['top5']:.1f}%  "
                       f"Top-10={eval_metrics['top10']:.1f}%  "
                       f"(n={eval_metrics['total']})")
            if 'threshold_at_1pct_far' in eval_metrics:
                logger.info(f"  Calibrated threshold @ 1% FAR: "
                           f"{eval_metrics['threshold_at_1pct_far']:.4f} "
                           f"(FRR={eval_metrics['frr_at_1pct_far']:.1f}%)")
        logger.info(f"  Time:  {train_metrics['time_minutes']:.1f} min")
        logger.info(f"  Best:  Top-1={best_top1:.1f}% (epoch {best_epoch})")
        logger.info("=" * 70)
        
        # Save best model based on identification Top-1
        current_top1 = eval_metrics['top1'] if eval_metrics else val_metrics['arcface_acc']
        
        if current_top1 > best_top1:
            best_top1 = current_top1
            best_epoch = epoch
            no_improve_count = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.to_dict(),
                'num_classes': num_classes,
                'class_to_label': train_dataset.class_to_label,
                'best_top1': best_top1,
                'eval_metrics': eval_metrics,
                'val_metrics': val_metrics,
            }
            if eval_metrics and 'threshold_at_1pct_far' in eval_metrics:
                checkpoint['calibrated_threshold'] = eval_metrics['threshold_at_1pct_far']
            
            torch.save(checkpoint, save_path / "best_model.pth")
            logger.info(f"  *** NEW BEST MODEL SAVED (Top-1: {best_top1:.1f}%) ***")
        else:
            no_improve_count += 1
        
        # Periodic checkpoint
        if epoch % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.to_dict(),
                'num_classes': num_classes,
                'class_to_label': train_dataset.class_to_label,
            }, save_path / f"checkpoint_epoch_{epoch}.pth")
        
        # Save history
        with open(save_path / "history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        # Early stopping
        if no_improve_count >= config.patience:
            logger.warning(f"\nEarly stopping: no improvement for {config.patience} epochs")
            break
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best Top-1 accuracy: {best_top1:.2f}% at epoch {best_epoch}")
    logger.info(f"Best model saved to: {save_path / 'best_model.pth'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        import traceback
        print(f"Training failed: {e}")
        print(traceback.format_exc())
        raise