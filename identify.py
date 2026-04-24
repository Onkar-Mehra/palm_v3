"""
Palm Biometric Registration & Identification System
====================================================
The main user-facing API. Two main operations:

1. REGISTER - Enroll a new person:
       palm_system.register(name="John Doe", rgb_path=..., ir_path=...)

2. IDENTIFY - Identify a person from new images:
       result = palm_system.identify(rgb_path=..., ir_path=...)
       print(result.name, result.confidence)

Key design points:
- Registered people are stored in a JSON database (their embedding + name).
- Identification finds the closest match in the database via cosine similarity.
- Matches below confidence threshold return "unknown" (rejection).
- Test-Time Augmentation (TTA) averages embeddings over multiple views.
- Works with people NOT in the original training set (generalizes via embedding).

Database file: enrolled_database/database.pkl
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# ============================================================
# RESULT CLASS
# ============================================================
class IdentificationResult:
    """Result of an identification attempt."""
    
    def __init__(self, name, confidence, top_k_matches=None, status='accepted'):
        self.name = name
        self.confidence = confidence  # cosine similarity in [0, 1]
        self.top_k_matches = top_k_matches or []  # list of (name, score)
        self.status = status  # 'accepted', 'rejected_low_confidence', 'no_database', 'no_hand'
    
    def __repr__(self):
        return (f"IdentificationResult(name='{self.name}', "
                f"confidence={self.confidence:.4f}, status='{self.status}')")
    
    def __str__(self):
        if self.status == 'accepted':
            s = f"IDENTIFIED: {self.name}\n"
            s += f"Confidence: {self.confidence:.4f}\n"
            if self.top_k_matches:
                s += f"\nTop matches:\n"
                for i, (name, score) in enumerate(self.top_k_matches, 1):
                    s += f"  {i}. {name}: {score:.4f}\n"
            return s
        elif self.status == 'rejected_low_confidence':
            s = f"REJECTED: confidence too low\n"
            s += f"Best guess was: {self.name} ({self.confidence:.4f})\n"
            if self.top_k_matches:
                s += f"\nAll candidates were below threshold. Top matches:\n"
                for i, (name, score) in enumerate(self.top_k_matches, 1):
                    s += f"  {i}. {name}: {score:.4f}\n"
            return s
        elif self.status == 'no_hand':
            return "ERROR: No hand detected in image"
        elif self.status == 'no_database':
            return "ERROR: No people enrolled in database"
        return f"Status: {self.status}"


# ============================================================
# THE SYSTEM
# ============================================================
class PalmBiometricSystem:
    """
    Main palm biometric system. Handles registration and identification.
    """
    
    def __init__(
        self,
        model_path: str,
        database_path: str = "enrolled_database/database.pkl",
        device: str = "cpu",
        match_threshold: Optional[float] = None,
        use_tta: bool = True,
    ):
        """
        Args:
            model_path: path to trained .pth file
            database_path: path to JSON database of enrolled people
            device: 'cpu' or 'cuda'
            match_threshold: cosine similarity threshold for accepting a match.
                            If None, uses calibrated value from training.
            use_tta: whether to use test-time augmentation
        """
        self.device = device
        self.database_path = Path(database_path)
        self.use_tta = use_tta
        
        # Lazy import to keep startup fast for help/usage
        from config import Config
        from model import PalmBiometricModel
        
        # Load checkpoint
        logger.info(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        config_dict = checkpoint['config']
        # Rebuild Config from dict
        self.config = Config()
        for k, v in config_dict.items():
            if hasattr(self.config, k):
                if isinstance(getattr(self.config, k), tuple) and isinstance(v, list):
                    v = tuple(v)
                setattr(self.config, k, v)
        
        self.config.device = device
        
        # Build model
        self.num_train_classes = checkpoint.get('num_classes', 500)
        self.model = PalmBiometricModel(self.config, self.num_train_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Threshold: explicit > calibrated > default
        if match_threshold is not None:
            self.match_threshold = float(match_threshold)
        elif 'calibrated_threshold' in checkpoint:
            self.match_threshold = float(checkpoint['calibrated_threshold'])
        else:
            self.match_threshold = self.config.default_match_threshold
        
        logger.info(f"Match threshold: {self.match_threshold:.4f}")
        
        # ROI extractor (lazy init)
        self._roi_extractor = None
        
        # Database (loaded on demand)
        self.database: Dict[str, np.ndarray] = {}
        self._load_database()
    
    @property
    def roi_extractor(self):
        """Lazy-init the ROI extractor."""
        if self._roi_extractor is None:
            from palm_roi import PalmROIExtractor
            self._roi_extractor = PalmROIExtractor(
                output_size=self.config.image_size[0]
            )
        return self._roi_extractor
    
    # ============================================================
    # DATABASE I/O
    # ============================================================
    def _load_database(self):
        """Load enrolled people database from disk."""
        if self.database_path.exists():
            with open(self.database_path, 'rb') as f:
                self.database = pickle.load(f)
            logger.info(f"Loaded database with {len(self.database)} enrolled people")
        else:
            self.database = {}
            logger.info("No existing database. Starting fresh.")
    
    def _save_database(self):
        """Persist database to disk."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.database, f)
        logger.info(f"Database saved to {self.database_path}")
    
    def list_enrolled(self) -> List[str]:
        """Return list of enrolled person names."""
        return sorted(self.database.keys())
    
    def num_enrolled(self) -> int:
        return len(self.database)
    
    def is_enrolled(self, name: str) -> bool:
        return name in self.database
    
    def remove_enrolled(self, name: str) -> bool:
        """Remove a person from the database."""
        if name in self.database:
            del self.database[name]
            self._save_database()
            logger.info(f"Removed: {name}")
            return True
        return False
    
    # ============================================================
    # IMAGE PREPROCESSING
    # ============================================================
    def _preprocess(self, rgb_img, ir_img, apply_roi=True):
        """
        Apply ROI extraction (if requested) + normalization.
        Returns (rgb_tensor, ir_tensor) both as 1×3×H×W.
        Returns (None, None) if hand cannot be detected.
        """
        if apply_roi:
            result = self.roi_extractor.extract_pair(rgb_img, ir_img)
            if result is None:
                return None, None
            rgb_img, ir_img = result
        else:
            # Just resize
            h, w = self.config.image_size
            rgb_img = cv2.resize(rgb_img, (w, h))
            ir_img = cv2.resize(ir_img, (w, h))
        
        # Use the validation augmenter for clean preprocessing
        from dataset import PairedAugmentation
        augmenter = PairedAugmentation(self.config, is_training=False)
        rgb_t, ir_t = augmenter(rgb_img, ir_img)
        
        return rgb_t.unsqueeze(0), ir_t.unsqueeze(0)
    
    @torch.no_grad()
    def _compute_embedding(self, rgb_path, ir_path, use_tta=None):
        """
        Compute embedding for a single (rgb, ir) image pair.
        If use_tta is True, averages embeddings over augmented versions.
        Returns numpy array of shape (embedding_dim,) or None.
        """
        if use_tta is None:
            use_tta = self.use_tta
        
        # Load images
        rgb = cv2.imread(str(rgb_path))
        ir = cv2.imread(str(ir_path))
        
        if rgb is None:
            raise IOError(f"Cannot read RGB image: {rgb_path}")
        if ir is None:
            raise IOError(f"Cannot read IR image: {ir_path}")
        
        # ROI + normalize
        rgb_t, ir_t = self._preprocess(rgb, ir, apply_roi=True)
        if rgb_t is None:
            return None  # No hand detected
        
        rgb_t = rgb_t.to(self.device)
        ir_t = ir_t.to(self.device)
        
        if not use_tta:
            emb = self.model.get_embedding(rgb_t, ir_t)
            return emb.cpu().numpy().squeeze()
        
        # TTA: original + horizontal flip + small rotations
        embeddings = []
        
        # 1. Original
        emb = self.model.get_embedding(rgb_t, ir_t)
        embeddings.append(emb)
        
        # 2. Horizontal flip
        emb = self.model.get_embedding(torch.flip(rgb_t, dims=[3]),
                                       torch.flip(ir_t, dims=[3]))
        embeddings.append(emb)
        
        # 3-5. Small rotations on the source image
        for angle in [-7, 7, 3]:
            rgb_np = cv2.imread(str(rgb_path))
            ir_np = cv2.imread(str(ir_path))
            
            roi_result = self.roi_extractor.extract_pair(rgb_np, ir_np)
            if roi_result is None:
                continue
            rgb_roi, ir_roi = roi_result
            
            h, w = rgb_roi.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rgb_rot = cv2.warpAffine(rgb_roi, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            ir_rot = cv2.warpAffine(ir_roi, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            from dataset import PairedAugmentation
            aug = PairedAugmentation(self.config, is_training=False)
            r_t, i_t = aug(rgb_rot, ir_rot)
            r_t = r_t.unsqueeze(0).to(self.device)
            i_t = i_t.unsqueeze(0).to(self.device)
            emb = self.model.get_embedding(r_t, i_t)
            embeddings.append(emb)
        
        # Average and re-normalize
        avg = torch.stack(embeddings).mean(dim=0)
        avg = F.normalize(avg, p=2, dim=1)
        return avg.cpu().numpy().squeeze()
    
    # ============================================================
    # REGISTRATION
    # ============================================================
    def register(
        self,
        name: str,
        rgb_path: str,
        ir_path: str,
        overwrite: bool = False,
        verbose: bool = True
    ) -> bool:
        """
        Register a new person.
        
        Args:
            name: unique identifier (person's name or ID)
            rgb_path: path to RGB palm image
            ir_path: path to IR palm image
            overwrite: if True, replaces existing entry; if False, refuses to overwrite
            verbose: print progress
        
        Returns:
            True if successful, False otherwise.
        """
        name = name.strip()
        if not name:
            logger.error("Name cannot be empty")
            return False
        
        if name in self.database and not overwrite:
            logger.error(f"'{name}' is already enrolled. Use overwrite=True to replace.")
            return False
        
        if verbose:
            logger.info(f"Registering: {name}")
            logger.info(f"  RGB: {rgb_path}")
            logger.info(f"  IR:  {ir_path}")
        
        # Compute embedding
        try:
            embedding = self._compute_embedding(rgb_path, ir_path, use_tta=self.use_tta)
        except IOError as e:
            logger.error(f"Failed to load images: {e}")
            return False
        
        if embedding is None:
            logger.error("Hand could not be detected in the image. Registration failed.")
            return False
        
        # Store in database
        self.database[name] = embedding.astype(np.float32)
        self._save_database()
        
        if verbose:
            logger.info(f"  ✓ Successfully enrolled (database now has {len(self.database)} people)")
        
        return True
    
    def register_multiple_images(
        self,
        name: str,
        image_pairs: List[Tuple[str, str]],
        overwrite: bool = False,
        verbose: bool = True,
    ) -> bool:
        """
        Register a person using multiple image pairs.
        Embeddings are averaged for a more robust prototype.
        
        Args:
            name: person's name
            image_pairs: list of (rgb_path, ir_path) tuples
        """
        name = name.strip()
        if not name:
            logger.error("Name cannot be empty")
            return False
        
        if name in self.database and not overwrite:
            logger.error(f"'{name}' already enrolled. Use overwrite=True.")
            return False
        
        embeddings = []
        for rgb_path, ir_path in image_pairs:
            try:
                emb = self._compute_embedding(rgb_path, ir_path, use_tta=self.use_tta)
                if emb is not None:
                    embeddings.append(emb)
                elif verbose:
                    logger.warning(f"  Skipped (no hand detected): {rgb_path}")
            except IOError as e:
                logger.warning(f"  Skipped (read error): {e}")
        
        if not embeddings:
            logger.error("No usable images for registration")
            return False
        
        # Mean and L2-normalize
        prototype = np.mean(embeddings, axis=0)
        prototype = prototype / (np.linalg.norm(prototype) + 1e-12)
        
        self.database[name] = prototype.astype(np.float32)
        self._save_database()
        
        if verbose:
            logger.info(f"  ✓ Enrolled '{name}' from {len(embeddings)} images")
        
        return True
    
    # ============================================================
    # IDENTIFICATION
    # ============================================================
    def identify(
        self,
        rgb_path: str,
        ir_path: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> IdentificationResult:
        """
        Identify a person from an RGB + IR image pair.
        
        Args:
            rgb_path: path to RGB query image
            ir_path: path to IR query image
            top_k: number of top candidates to return for inspection
            threshold: override the system's default match threshold
        
        Returns:
            IdentificationResult
        """
        if not self.database:
            return IdentificationResult(
                name="", confidence=0.0, status='no_database'
            )
        
        thresh = threshold if threshold is not None else self.match_threshold
        
        try:
            query_emb = self._compute_embedding(rgb_path, ir_path, use_tta=self.use_tta)
        except IOError as e:
            logger.error(f"Image read error: {e}")
            return IdentificationResult(
                name="", confidence=0.0, status='no_hand'
            )
        
        if query_emb is None:
            return IdentificationResult(
                name="", confidence=0.0, status='no_hand'
            )
        
        # Compute similarity to all enrolled people
        names = list(self.database.keys())
        gallery = np.stack([self.database[n] for n in names])  # (N, D)
        
        # Cosine similarity (both query and gallery are L2-normalized)
        similarities = gallery @ query_emb  # (N,)
        
        # Sort
        sorted_idx = np.argsort(-similarities)
        top_k_actual = min(top_k, len(names))
        top_matches = [
            (names[sorted_idx[i]], float(similarities[sorted_idx[i]]))
            for i in range(top_k_actual)
        ]
        
        best_name, best_score = top_matches[0]
        
        if best_score >= thresh:
            return IdentificationResult(
                name=best_name,
                confidence=best_score,
                top_k_matches=top_matches,
                status='accepted',
            )
        else:
            return IdentificationResult(
                name=best_name,
                confidence=best_score,
                top_k_matches=top_matches,
                status='rejected_low_confidence',
            )
    
    def verify(
        self,
        claimed_name: str,
        rgb_path: str,
        ir_path: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Verify that the person in the images matches the claimed identity.
        
        Returns:
            (is_match, confidence_score)
        """
        if claimed_name not in self.database:
            return False, 0.0
        
        thresh = threshold if threshold is not None else self.match_threshold
        
        try:
            query_emb = self._compute_embedding(rgb_path, ir_path, use_tta=self.use_tta)
        except IOError:
            return False, 0.0
        
        if query_emb is None:
            return False, 0.0
        
        gallery_emb = self.database[claimed_name]
        score = float(np.dot(query_emb, gallery_emb))
        
        return score >= thresh, score
    
    # ============================================================
    # BATCH ENROLLMENT FROM A FOLDER
    # ============================================================
    def enroll_from_folder(
        self,
        folder: str,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> Dict[str, bool]:
        """
        Enroll many people at once from a folder structure:
            folder/
                Alice/
                    rgb.jpg, ir.jpg
                Bob/
                    rgb.jpg, ir.jpg
                ...
        
        Each subfolder name becomes the person's name.
        Multiple image pairs per person are averaged.
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(folder)
        
        person_dirs = sorted([d for d in folder.iterdir() if d.is_dir()
                             and not d.name.startswith('_')])
        
        results = {}
        
        for person_dir in person_dirs:
            name = person_dir.name
            
            # Find all RGB and IR pairs
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
                if verbose:
                    logger.warning(f"  {name}: no rgb/ir files found, skipping")
                results[name] = False
                continue
            
            pairs = list(zip(rgb_files, ir_files))[:min(len(rgb_files), len(ir_files))]
            
            success = self.register_multiple_images(
                name, [(str(r), str(i)) for r, i in pairs],
                overwrite=overwrite, verbose=verbose
            )
            results[name] = success
        
        if verbose:
            success_count = sum(1 for v in results.values() if v)
            logger.info(f"\nBatch enrollment complete: {success_count}/{len(results)} successful")
        
        return results


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================
def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(
        description="Palm Biometric Registration & Identification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  # Register one person:
    python identify.py register --name "John Doe" \\
        --rgb path/to/john_rgb.jpg --ir path/to/john_ir.jpg

  # Register everyone from a folder (folder names = person names):
    python identify.py enroll_folder --folder enrolled_people/

  # Identify from a query image:
    python identify.py identify --rgb query_rgb.jpg --ir query_ir.jpg

  # Verify a claimed identity:
    python identify.py verify --name "John Doe" \\
        --rgb query_rgb.jpg --ir query_ir.jpg

  # List all enrolled people:
    python identify.py list

  # Remove someone:
    python identify.py remove --name "John Doe"
"""
    )
    
    parser.add_argument('command', choices=[
        'register', 'identify', 'verify', 'list', 'remove', 'enroll_folder'
    ])
    parser.add_argument('--model', default='models/best_model.pth',
                        help='Path to trained model (default: models/best_model.pth)')
    parser.add_argument('--database', default='enrolled_database/database.pkl',
                        help='Path to database file')
    parser.add_argument('--name', type=str, help='Person name')
    parser.add_argument('--rgb', type=str, help='Path to RGB image')
    parser.add_argument('--ir', type=str, help='Path to IR image')
    parser.add_argument('--folder', type=str, help='Folder for batch enrollment')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override match threshold (default: calibrated)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top candidates to show')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting existing entry')
    parser.add_argument('--no_tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    # Initialize system
    system = PalmBiometricSystem(
        model_path=args.model,
        database_path=args.database,
        device=args.device,
        match_threshold=args.threshold,
        use_tta=not args.no_tta,
    )
    
    # Dispatch
    if args.command == 'register':
        if not (args.name and args.rgb and args.ir):
            parser.error("register requires --name, --rgb, --ir")
        success = system.register(args.name, args.rgb, args.ir,
                                  overwrite=args.overwrite)
        sys.exit(0 if success else 1)
    
    elif args.command == 'identify':
        if not (args.rgb and args.ir):
            parser.error("identify requires --rgb and --ir")
        result = system.identify(args.rgb, args.ir, top_k=args.top_k)
        print()
        print("=" * 60)
        print(result)
        print("=" * 60)
        sys.exit(0 if result.status == 'accepted' else 1)
    
    elif args.command == 'verify':
        if not (args.name and args.rgb and args.ir):
            parser.error("verify requires --name, --rgb, --ir")
        is_match, score = system.verify(args.name, args.rgb, args.ir)
        print()
        print("=" * 60)
        if is_match:
            print(f"VERIFIED: This is {args.name} (score: {score:.4f})")
        else:
            print(f"REJECTED: This does not match {args.name} (score: {score:.4f})")
        print("=" * 60)
        sys.exit(0 if is_match else 1)
    
    elif args.command == 'list':
        names = system.list_enrolled()
        print(f"\nEnrolled people: {len(names)}")
        print("=" * 60)
        for name in names:
            print(f"  - {name}")
        print("=" * 60)
    
    elif args.command == 'remove':
        if not args.name:
            parser.error("remove requires --name")
        success = system.remove_enrolled(args.name)
        if success:
            print(f"Removed: {args.name}")
        else:
            print(f"Not found: {args.name}")
        sys.exit(0 if success else 1)
    
    elif args.command == 'enroll_folder':
        if not args.folder:
            parser.error("enroll_folder requires --folder")
        results = system.enroll_from_folder(args.folder, overwrite=args.overwrite)
        success = sum(1 for v in results.values() if v)
        print(f"\nEnrolled {success} of {len(results)} people")


if __name__ == "__main__":
    main()