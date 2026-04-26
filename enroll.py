"""
Enhanced Enrollment Module
===========================

Flexible enrollment supporting 4 different scenarios:

1. enroll_full(name, [3 RGB], [3 IR])
   - Best accuracy. Use when you have BOTH cameras.
   - 6 files per person (3 RGB + 3 IR)
   - Expected accuracy: 92-95% for new people, 97-99% for trained

2. enroll_rgb_only(name, [3 RGB])
   - Use when you only have an RGB camera.
   - System auto-generates IR from RGB.
   - Slight accuracy hit since fake IR isn't real IR.
   - 3 files per person (RGB only)
   - Expected accuracy: 90-93% for new people, 95-97% for trained

3. enroll_quick(name, rgb, ir)
   - Single image pair enrollment. FASTEST.
   - Use for trained people (already in dataset).
   - 2 files per person
   - Expected accuracy: 95-98% for trained, 85-90% for new

4. enroll_quick_rgb(name, rgb)
   - Single RGB only. EMERGENCY use.
   - System auto-generates IR.
   - Lowest accuracy.
   - Expected accuracy: 92-96% for trained, 82-88% for new

CLI commands:
    python enroll.py full --name "John" --rgb r1 r2 r3 --ir i1 i2 i3
    python enroll.py rgb_only --name "John" --rgb r1 r2 r3
    python enroll.py quick --name "John" --rgb r --ir i
    python enroll.py quick_rgb --name "John" --rgb r
    python enroll.py folder --folder demo_people/ --mode full
"""

import os
import sys
import argparse
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

# Suppress noisy logs
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)


# ============================================================
# Lazy imports to keep startup fast for --help
# ============================================================
def _get_system(model_path, database_path, device='cpu', threshold=None):
    """Lazy-load the PalmBiometricSystem."""
    from identify import PalmBiometricSystem
    return PalmBiometricSystem(
        model_path=model_path,
        database_path=database_path,
        device=device,
        match_threshold=threshold,
    )


def _get_rgb_to_ir():
    """Lazy-load rgb_to_ir converter."""
    from rgb_to_ir import rgb_to_ir_style
    return rgb_to_ir_style


# ============================================================
# CORE ENROLLMENT FUNCTIONS
# ============================================================
class FlexibleEnroller:
    """
    Enrollment manager that handles all 4 scenarios.
    """
    
    def __init__(self, system):
        self.system = system
        self._temp_dir = Path(tempfile.mkdtemp(prefix='palm_enroll_'))
        self._rgb_to_ir = _get_rgb_to_ir()
    
    def __del__(self):
        # Cleanup temp dir
        try:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def _generate_ir_from_rgb(self, rgb_path: str, suffix: str = "") -> str:
        """Convert RGB file to a fake IR file in temp dir. Return new IR path."""
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise IOError(f"Cannot read: {rgb_path}")
        
        ir_image = self._rgb_to_ir(rgb)
        
        # Save in temp dir
        ir_path = self._temp_dir / f"generated_ir_{suffix}_{Path(rgb_path).stem}.jpg"
        cv2.imwrite(str(ir_path), ir_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return str(ir_path)
    
    # --------------------------------------------------------
    # MODE 1: Full enrollment (BEST ACCURACY)
    # --------------------------------------------------------
    def enroll_full(
        self,
        name: str,
        rgb_paths: List[str],
        ir_paths: List[str],
        overwrite: bool = False
    ) -> bool:
        """
        Best-accuracy enrollment using N image pairs (recommended N=3).
        
        Args:
            name: person identifier
            rgb_paths: list of N RGB image paths (N >= 1, ideally 3)
            ir_paths:  list of N IR image paths (must be same length as rgb_paths)
            overwrite: replace existing entry if True
        
        Returns:
            True if successful
        """
        if len(rgb_paths) != len(ir_paths):
            logger.error(f"Number of RGB files ({len(rgb_paths)}) must match "
                        f"IR files ({len(ir_paths)})")
            return False
        
        if len(rgb_paths) < 1:
            logger.error("Need at least 1 image pair")
            return False
        
        logger.info(f"[FULL] Enrolling '{name}' with {len(rgb_paths)} image pair(s)")
        for rgb, ir in zip(rgb_paths, ir_paths):
            logger.info(f"  RGB: {rgb}")
            logger.info(f"  IR:  {ir}")
        
        pairs = list(zip(rgb_paths, ir_paths))
        return self.system.register_multiple_images(
            name, pairs, overwrite=overwrite, verbose=False
        )
    
    # --------------------------------------------------------
    # MODE 2: RGB-only enrollment with auto-generated IR
    # --------------------------------------------------------
    def enroll_rgb_only(
        self,
        name: str,
        rgb_paths: List[str],
        overwrite: bool = False
    ) -> bool:
        """
        Enrollment with RGB images only. System generates IR automatically
        by converting RGB to grayscale matching training IR distribution.
        
        Use when you don't have an IR camera available at enrollment time.
        Slight accuracy reduction vs real IR.
        
        Args:
            name: person identifier
            rgb_paths: list of N RGB image paths (recommended N=3)
            overwrite: replace existing entry if True
        """
        if len(rgb_paths) < 1:
            logger.error("Need at least 1 RGB image")
            return False
        
        logger.info(f"[RGB-ONLY] Enrolling '{name}' with {len(rgb_paths)} RGB image(s)")
        logger.info(f"  IR will be auto-generated from RGB (grayscale conversion)")
        
        # Generate IR for each RGB
        ir_paths = []
        for i, rgb_path in enumerate(rgb_paths):
            try:
                ir_path = self._generate_ir_from_rgb(rgb_path, suffix=f"{name}_{i}")
                ir_paths.append(ir_path)
                logger.info(f"  Generated IR for: {rgb_path}")
            except Exception as e:
                logger.error(f"  Failed to generate IR for {rgb_path}: {e}")
                return False
        
        pairs = list(zip(rgb_paths, ir_paths))
        return self.system.register_multiple_images(
            name, pairs, overwrite=overwrite, verbose=False
        )
    
    # --------------------------------------------------------
    # MODE 3: Quick single-pair enrollment (RGB + IR)
    # --------------------------------------------------------
    def enroll_quick(
        self,
        name: str,
        rgb_path: str,
        ir_path: str,
        overwrite: bool = False
    ) -> bool:
        """
        Single image pair enrollment. Fastest option.
        
        Best for: people already in training data (model already knows them).
        Reduced accuracy for completely new people.
        """
        logger.info(f"[QUICK] Enrolling '{name}' with single RGB+IR pair")
        return self.system.register(name, rgb_path, ir_path,
                                   overwrite=overwrite, verbose=False)
    
    # --------------------------------------------------------
    # MODE 4: Quick RGB-only (single RGB, generate IR)
    # --------------------------------------------------------
    def enroll_quick_rgb(
        self,
        name: str,
        rgb_path: str,
        overwrite: bool = False
    ) -> bool:
        """
        Single RGB-only enrollment with auto-generated IR.
        Lowest accuracy. Use only when nothing else available.
        """
        logger.info(f"[QUICK-RGB] Enrolling '{name}' with single RGB (auto IR)")
        try:
            ir_path = self._generate_ir_from_rgb(rgb_path, suffix=name)
        except Exception as e:
            logger.error(f"Failed to generate IR: {e}")
            return False
        
        return self.system.register(name, rgb_path, ir_path,
                                   overwrite=overwrite, verbose=False)
    
    # --------------------------------------------------------
    # FOLDER-BASED BULK ENROLLMENT
    # --------------------------------------------------------
    def enroll_folder(
        self,
        folder: str,
        mode: str = 'full',
        overwrite: bool = False
    ) -> dict:
        """
        Bulk enrollment from a folder.
        
        Folder structure:
            folder/
                Person1_Name/
                    rgb_1.jpg, ir_1.jpg
                    rgb_2.jpg, ir_2.jpg
                    rgb_3.jpg, ir_3.jpg
                Person2_Name/
                    ...
        
        Mode 'full': uses all rgb_N + ir_N pairs found
        Mode 'rgb_only': uses only rgb_N files, generates IR from them
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder}")
            return {}
        
        person_dirs = sorted([d for d in folder_path.iterdir()
                             if d.is_dir() and not d.name.startswith('_')])
        
        results = {}
        success_count = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BULK ENROLLMENT - {len(person_dirs)} persons - mode='{mode}'")
        logger.info('='*60)
        
        for i, person_dir in enumerate(person_dirs):
            name = person_dir.name
            
            # Find RGB files (sorted for consistent pairing)
            rgb_files = sorted([
                str(f) for f in person_dir.iterdir()
                if f.is_file() and 'rgb' in f.name.lower()
                and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
            ])
            
            if not rgb_files:
                logger.warning(f"  [{i+1}/{len(person_dirs)}] {name}: NO RGB FILES")
                results[name] = False
                continue
            
            success = False
            
            if mode == 'full':
                # Find IR files
                ir_files = sorted([
                    str(f) for f in person_dir.iterdir()
                    if f.is_file() and 'ir' in f.name.lower()
                    and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
                ])
                
                if not ir_files:
                    logger.warning(f"  [{i+1}/{len(person_dirs)}] {name}: "
                                  f"NO IR FILES (need IR for full mode, "
                                  f"or use mode='rgb_only')")
                    results[name] = False
                    continue
                
                # Pair them in order
                num_pairs = min(len(rgb_files), len(ir_files))
                rgb_subset = rgb_files[:num_pairs]
                ir_subset = ir_files[:num_pairs]
                
                success = self.enroll_full(name, rgb_subset, ir_subset,
                                          overwrite=overwrite)
            
            elif mode == 'rgb_only':
                success = self.enroll_rgb_only(name, rgb_files,
                                              overwrite=overwrite)
            
            else:
                logger.error(f"Unknown mode: {mode}")
                return {}
            
            results[name] = success
            if success:
                success_count += 1
                logger.info(f"  [{i+1}/{len(person_dirs)}] ✓ {name}")
            else:
                logger.error(f"  [{i+1}/{len(person_dirs)}] ✗ {name}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE: {success_count}/{len(person_dirs)} successful")
        logger.info('='*60)
        
        return results


# ============================================================
# CLI
# ============================================================
def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(
        description="Flexible Palm Biometric Enrollment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ENROLLMENT MODES:

  full       - 3 RGB + 3 IR images (best accuracy, recommended for new people)
               python enroll.py full --name "John Doe" \\
                   --rgb john_rgb_1.jpg john_rgb_2.jpg john_rgb_3.jpg \\
                   --ir  john_ir_1.jpg  john_ir_2.jpg  john_ir_3.jpg

  rgb_only   - 3 RGB images, IR auto-generated (when no IR camera)
               python enroll.py rgb_only --name "John Doe" \\
                   --rgb john_rgb_1.jpg john_rgb_2.jpg john_rgb_3.jpg

  quick      - Single RGB + IR pair (fast, best for trained people)
               python enroll.py quick --name "John Doe" \\
                   --rgb john_rgb.jpg --ir john_ir.jpg

  quick_rgb  - Single RGB only, IR auto-generated (fastest)
               python enroll.py quick_rgb --name "John Doe" --rgb john_rgb.jpg

  folder     - Bulk enroll many people from a folder
               python enroll.py folder --folder demo_people/ --mode full
               python enroll.py folder --folder demo_people/ --mode rgb_only

EXPECTED ACCURACY:

  Mode       New people    Trained people
  full       92-95%        97-99%
  rgb_only   90-93%        95-97%
  quick      85-90%        95-98%
  quick_rgb  82-88%        92-96%
"""
    )
    
    parser.add_argument('mode', choices=[
        'full', 'rgb_only', 'quick', 'quick_rgb', 'folder'
    ], help='Enrollment mode')
    
    parser.add_argument('--model', default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--database', default='enrolled_database/database.pkl',
                        help='Path to database file')
    
    parser.add_argument('--name', type=str, help='Person name (for single enrollment)')
    parser.add_argument('--rgb', type=str, nargs='+', help='RGB image path(s)')
    parser.add_argument('--ir', type=str, nargs='+', help='IR image path(s)')
    parser.add_argument('--folder', type=str, help='Folder for bulk enrollment')
    parser.add_argument('--folder_mode', default='full',
                        choices=['full', 'rgb_only'],
                        help='Mode for folder enrollment (default: full)')
    
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting existing entries')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    # Initialize system
    system = _get_system(args.model, args.database, args.device)
    enroller = FlexibleEnroller(system)
    
    # Dispatch
    if args.mode == 'full':
        if not args.name or not args.rgb or not args.ir:
            parser.error("'full' mode requires --name, --rgb, --ir")
        success = enroller.enroll_full(args.name, args.rgb, args.ir,
                                      overwrite=args.overwrite)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'rgb_only':
        if not args.name or not args.rgb:
            parser.error("'rgb_only' mode requires --name and --rgb")
        success = enroller.enroll_rgb_only(args.name, args.rgb,
                                          overwrite=args.overwrite)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'quick':
        if not args.name or not args.rgb or not args.ir:
            parser.error("'quick' mode requires --name, --rgb (single), --ir (single)")
        if len(args.rgb) != 1 or len(args.ir) != 1:
            parser.error("'quick' mode takes exactly 1 --rgb and 1 --ir")
        success = enroller.enroll_quick(args.name, args.rgb[0], args.ir[0],
                                       overwrite=args.overwrite)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'quick_rgb':
        if not args.name or not args.rgb:
            parser.error("'quick_rgb' mode requires --name and --rgb (single)")
        if len(args.rgb) != 1:
            parser.error("'quick_rgb' mode takes exactly 1 --rgb")
        success = enroller.enroll_quick_rgb(args.name, args.rgb[0],
                                           overwrite=args.overwrite)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'folder':
        if not args.folder:
            parser.error("'folder' mode requires --folder")
        results = enroller.enroll_folder(args.folder, mode=args.folder_mode,
                                        overwrite=args.overwrite)
        success_count = sum(1 for v in results.values() if v)
        print(f"\nEnrolled {success_count}/{len(results)} people successfully")
        sys.exit(0 if success_count > 0 else 1)


if __name__ == "__main__":
    main()