"""
Palm Biometric Model
=====================
Architecture:
  - Dual-stream encoder: RGB ResNet50 + IR ResNet50 (both pretrained)
  - Concatenation + projection to embedding
  - L2-normalized embeddings (for cosine similarity)
  - ArcFace head for training (margin-based softmax)
  - Standard linear head for cross-entropy (joint training)

Why this beats prototypical networks for fixed populations:
  - Direct supervision on identity (cross-entropy)
  - ArcFace creates well-separated angular clusters
  - L2-normalized embeddings work cleanly with cosine similarity
  - 500-class classifier > prototypical for known fixed people
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================
# DUAL STREAM ENCODER
# ============================================================
class DualStreamEncoder(nn.Module):
    """
    Two ResNet50 backbones (one for RGB, one for IR-tiled-to-3-channels).
    Outputs L2-normalized embedding.
    """
    
    def __init__(self, embedding_dim=512, dropout=0.4, pretrained=True):
        super().__init__()
        
        from torchvision.models import resnet50, ResNet50_Weights
        
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        
        # RGB stream
        self.rgb_backbone = resnet50(weights=weights)
        rgb_feat_dim = self.rgb_backbone.fc.in_features  # 2048
        self.rgb_backbone.fc = nn.Identity()
        
        # IR stream (uses pretrained ImageNet weights too; IR is tiled to 3ch)
        self.ir_backbone = resnet50(weights=weights)
        self.ir_backbone.fc = nn.Identity()
        
        # Fusion
        fused_dim = rgb_feat_dim * 2  # 4096
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, rgb, ir):
        rgb_feat = self.rgb_backbone(rgb)   # (B, 2048)
        ir_feat = self.ir_backbone(ir)       # (B, 2048)
        fused = torch.cat([rgb_feat, ir_feat], dim=1)  # (B, 4096)
        emb = self.fusion(fused)             # (B, embedding_dim)
        # L2 normalize
        emb = F.normalize(emb, p=2, dim=1)
        return emb
    
    def freeze_backbones(self):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = False
        for p in self.ir_backbone.parameters():
            p.requires_grad = False
    
    def unfreeze_backbones(self):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = True
        for p in self.ir_backbone.parameters():
            p.requires_grad = True


# ============================================================
# ARCFACE HEAD
# ============================================================
class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    
    Adds an angular margin between the query embedding and its true class
    weight vector, forcing the model to learn more discriminative embeddings.
    
    During training: applies the margin penalty.
    During inference: behaves like a normal classification head.
    """
    
    def __init__(self, embedding_dim, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        
        # Class weight matrix (num_classes, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        
        # Precomputed for margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, labels=None):
        """
        Args:
            embeddings: (B, embedding_dim) - already L2 normalized
            labels: (B,) - if provided, applies arcface margin
        Returns:
            logits: (B, num_classes) - scaled and margin-adjusted
        """
        # Normalize weights
        W = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity
        cosine = F.linear(embeddings, W)  # (B, num_classes)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        if labels is None:
            # Inference mode - return scaled cosines
            return cosine * self.scale
        
        # Training mode - apply margin to true class
        sine = torch.sqrt(1.0 - cosine ** 2)
        # cos(theta + margin) = cos(theta)*cos(m) - sin(theta)*sin(m)
        cos_theta_plus_m = cosine * self.cos_m - sine * self.sin_m
        
        # Numerical stability: only apply margin where cos(theta) > threshold
        cos_theta_plus_m = torch.where(
            cosine > self.threshold,
            cos_theta_plus_m,
            cosine - self.mm
        )
        
        # One-hot encode labels and replace true-class cosine with margin-adjusted
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        logits = (one_hot * cos_theta_plus_m) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale
        
        return logits


# ============================================================
# COMPLETE MODEL
# ============================================================
class PalmBiometricModel(nn.Module):
    """
    Full model: encoder + ArcFace head + standard CE head.
    
    Joint training: total_loss = w1*arcface_loss + w2*ce_loss
    """
    
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = DualStreamEncoder(
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            pretrained=config.pretrained,
        )
        
        # ArcFace head (for metric learning)
        self.arcface_head = ArcFaceHead(
            embedding_dim=config.embedding_dim,
            num_classes=num_classes,
            scale=config.arcface_scale,
            margin=config.arcface_margin,
        )
        
        # Standard linear head (for cross-entropy)
        # NOTE: no bias and weights normalized to keep similar geometry to ArcFace
        self.ce_head = nn.Linear(config.embedding_dim, num_classes, bias=True)
        
        # For inference: storage of enrolled prototypes (for new people not in training)
        # NOT used during training - only for adding new people post-training
        self.register_buffer('_dummy', torch.zeros(1))
    
    def forward(self, rgb, ir, labels=None):
        """
        Returns dict with:
            embeddings: (B, embedding_dim) L2-normalized
            arcface_logits: (B, num_classes)
            ce_logits: (B, num_classes)
        """
        emb = self.encoder(rgb, ir)
        arcface_logits = self.arcface_head(emb, labels)
        ce_logits = self.ce_head(emb)
        return {
            'embeddings': emb,
            'arcface_logits': arcface_logits,
            'ce_logits': ce_logits,
        }
    
    def get_embedding(self, rgb, ir):
        """Get just the embedding (for inference / matching)."""
        return self.encoder(rgb, ir)
    
    def freeze_backbones(self):
        self.encoder.freeze_backbones()
    
    def unfreeze_backbones(self):
        self.encoder.unfreeze_backbones()


# ============================================================
# COMBINED LOSS
# ============================================================
class JointLoss(nn.Module):
    """Joint ArcFace + Cross-Entropy loss."""
    
    def __init__(self, config):
        super().__init__()
        self.weight_arcface = config.weight_arcface
        self.weight_ce = config.weight_ce
        self.label_smoothing = config.label_smoothing
        
        self.arcface_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    def forward(self, outputs, labels):
        af_loss = self.arcface_loss(outputs['arcface_logits'], labels)
        ce_loss = self.ce_loss(outputs['ce_logits'], labels)
        
        total_loss = self.weight_arcface * af_loss + self.weight_ce * ce_loss
        
        # Compute accuracies for logging
        with torch.no_grad():
            af_preds = outputs['arcface_logits'].argmax(dim=1)
            ce_preds = outputs['ce_logits'].argmax(dim=1)
            af_acc = (af_preds == labels).float().mean().item() * 100
            ce_acc = (ce_preds == labels).float().mean().item() * 100
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'arcface_loss': af_loss.item(),
            'ce_loss': ce_loss.item(),
            'arcface_acc': af_acc,
            'ce_acc': ce_acc,
        }