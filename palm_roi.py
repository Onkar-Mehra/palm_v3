import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, List
import argparse
import time


# MediaPipe hand landmark indices we use:
#   0  = wrist
#   5  = base of index finger
#   9  = base of middle finger
#  17  = base of pinky finger
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


class PalmROIExtractor:
    """Extracts a canonical, aligned palm ROI from a hand image."""
    
    def __init__(self, output_size=224, min_detection_confidence=0.3, margin_factor=1.6):
        self.output_size = output_size
        self.margin_factor = margin_factor
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            model_complexity=1,
        )
    
    def _get_landmarks(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        if not result.multi_hand_landmarks:
            return None
        landmarks = result.multi_hand_landmarks[0].landmark
        coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        return coords
    
    def _compute_transform(self, landmarks):
        wrist = landmarks[WRIST]
        index_mcp = landmarks[INDEX_MCP]
        middle_mcp = landmarks[MIDDLE_MCP]
        pinky_mcp = landmarks[PINKY_MCP]
        
        palm_center = (wrist + middle_mcp) / 2.0
        up_vec = middle_mcp - wrist
        angle_rad = np.arctan2(up_vec[0], -up_vec[1])
        
        palm_width = np.linalg.norm(index_mcp - pinky_mcp)
        palm_height = np.linalg.norm(middle_mcp - wrist)
        palm_scale = max(palm_width, palm_height) * self.margin_factor
        
        if palm_scale < 20:
            return None
        
        scale_factor = self.output_size / palm_scale
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        cx, cy = palm_center
        out_c = self.output_size / 2.0
        
        M = np.array([
            [scale_factor * cos_a, -scale_factor * sin_a,
             out_c - scale_factor * (cos_a * cx - sin_a * cy)],
            [scale_factor * sin_a,  scale_factor * cos_a,
             out_c - scale_factor * (sin_a * cx + cos_a * cy)]
        ], dtype=np.float32)
        return M
    
    def extract(self, image_bgr):
        """Extract palm ROI from a single image. Returns ROI or None."""
        landmarks = self._get_landmarks(image_bgr)
        if landmarks is None:
            return None
        M = self._compute_transform(landmarks)
        if M is None:
            return None
        return cv2.warpAffine(
            image_bgr, M, (self.output_size, self.output_size),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
        )
    
    def extract_pair(self, rgb_bgr, ir_bgr):
        """
        Extract aligned ROIs from RGB and IR.
        Tries RGB first; falls back to IR if RGB detection fails.
        Returns (rgb_roi, ir_roi) or None.
        """
        if rgb_bgr is None or ir_bgr is None:
            return None
        
        landmarks = self._get_landmarks(rgb_bgr)
        landmarks_from = "rgb"
        if landmarks is None:
            landmarks = self._get_landmarks(ir_bgr)
            landmarks_from = "ir"
        if landmarks is None:
            return None
        
        rgb_h, rgb_w = rgb_bgr.shape[:2]
        ir_h, ir_w = ir_bgr.shape[:2]
        
        if landmarks_from == "rgb":
            M_rgb = self._compute_transform(landmarks)
            if M_rgb is None:
                return None
            if (ir_h, ir_w) != (rgb_h, rgb_w):
                ir_landmarks = landmarks * np.array([ir_w/rgb_w, ir_h/rgb_h], dtype=np.float32)
                M_ir = self._compute_transform(ir_landmarks)
                if M_ir is None:
                    M_ir = M_rgb
            else:
                M_ir = M_rgb
        else:
            M_ir = self._compute_transform(landmarks)
            if M_ir is None:
                return None
            if (rgb_h, rgb_w) != (ir_h, ir_w):
                rgb_landmarks = landmarks * np.array([rgb_w/ir_w, rgb_h/ir_h], dtype=np.float32)
                M_rgb = self._compute_transform(rgb_landmarks)
                if M_rgb is None:
                    M_rgb = M_ir
            else:
                M_rgb = M_ir
        
        rgb_roi = cv2.warpAffine(
            rgb_bgr, M_rgb, (self.output_size, self.output_size),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
        )
        ir_roi = cv2.warpAffine(
            ir_bgr, M_ir, (self.output_size, self.output_size),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
        )
        return rgb_roi, ir_roi
    
    def close(self):
        self.hands.close()


def _center_crop_resize(img, size):
    """Fallback when MediaPipe can't detect a hand."""
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return cv2.resize(img[y0:y0+s, x0:x0+s], (size, size), interpolation=cv2.INTER_AREA)


def find_image_files(person_dir):
    """Find one RGB and one IR image in a person folder."""
    all_files = list(person_dir.iterdir())
    rgb_path, ir_path = None, None
    for f in all_files:
        if not f.is_file() or f.name.startswith('.'):
            continue
        name = f.name.lower()
        if 'ir' in name and ir_path is None:
            ir_path = f
        elif 'rgb' in name and rgb_path is None:
            rgb_path = f
    if rgb_path is None or ir_path is None:
        image_files = sorted([
            f for f in all_files
            if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ])
        if len(image_files) >= 2:
            if rgb_path is None:
                rgb_path = image_files[0]
            if ir_path is None:
                ir_path = image_files[1] if image_files[1] != rgb_path else image_files[0]
    return rgb_path, ir_path


def preprocess_dataset(input_dir, output_dir, output_size=224, margin_factor=1.6):
    """Preprocess an entire palm dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    
    extractor = PalmROIExtractor(output_size=output_size, margin_factor=margin_factor)
    person_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    total = len(person_dirs)
    
    if total == 0:
        raise ValueError(f"No person folders found in {input_dir}")
    
    print(f"Found {total} person folders in {input_dir}")
    print(f"Output size: {output_size}x{output_size}")
    print(f"Writing to: {output_dir}\n")
    
    detected, fallback, failed = 0, 0, 0
    failures = []
    start_time = time.time()
    
    for i, person_dir in enumerate(person_dirs):
        person_name = person_dir.name
        out_person = output_path / person_name
        out_person.mkdir(parents=True, exist_ok=True)
        
        rgb_path, ir_path = find_image_files(person_dir)
        if rgb_path is None or ir_path is None:
            failed += 1
            failures.append((person_name, "missing rgb or ir file"))
            continue
        
        rgb = cv2.imread(str(rgb_path))
        ir = cv2.imread(str(ir_path))
        if rgb is None or ir is None:
            failed += 1
            failures.append((person_name, "could not read images"))
            continue
        
        result = extractor.extract_pair(rgb, ir)
        if result is not None:
            rgb_roi, ir_roi = result
            detected += 1
        else:
            rgb_roi = _center_crop_resize(rgb, output_size)
            ir_roi = _center_crop_resize(ir, output_size)
            fallback += 1
            failures.append((person_name, "no hand detected - center crop used"))
        
        cv2.imwrite(str(out_person / "rgb.jpg"), rgb_roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_person / "ir.jpg"), ir_roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1:>5}/{total}  |  detected: {detected}  fallback: {fallback}  "
                  f"failed: {failed}  |  {rate:.1f} img/s  |  ETA: {eta:.0f}s")
    
    extractor.close()
    elapsed = time.time() - start_time
    
    print()
    print("=" * 60)
    print("ROI Extraction Complete")
    print("=" * 60)
    print(f"Total persons:        {total}")
    print(f"Hand detected (good): {detected} ({100*detected/total:.1f}%)")
    print(f"Center-crop fallback: {fallback} ({100*fallback/total:.1f}%)")
    print(f"Completely failed:    {failed} ({100*failed/total:.1f}%)")
    print(f"Total time:           {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)
    
    log_path = output_path / "_roi_extraction_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Total: {total}\nDetected: {detected}\n")
        f.write(f"Fallback: {fallback}\nFailed: {failed}\n\n")
        f.write("Person\tStatus\n")
        for name, reason in failures:
            f.write(f"{name}\t{reason}\n")
    print(f"\nLog written to: {log_path}")
    
    return {'total': total, 'detected': detected, 'fallback': fallback,
            'failed': failed, 'failures': failures}


def main():
    parser = argparse.ArgumentParser(
        description="Extract palm ROIs from hand images using MediaPipe."
    )
    parser.add_argument('--input', required=True, help='Input dir (e.g., final_folder)')
    parser.add_argument('--output', required=True, help='Output dir for ROI images')
    parser.add_argument('--size', type=int, default=224, help='Output size (default: 224)')
    parser.add_argument('--margin', type=float, default=1.6,
                        help='Palm margin (1.2=tight, 1.6=default, 2.0=loose)')
    args = parser.parse_args()
    
    preprocess_dataset(args.input, args.output, args.size, args.margin)


if __name__ == "__main__":
    main()