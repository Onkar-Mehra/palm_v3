"""
INFERENCE SCRIPT
=================
Use trained model for classification.

Usage:
    python inference.py --model models_fewshot/best_model.pth --data_dir final_folder
    python inference.py --model models_fewshot/best_model.pth --rgb test_rgb.jpg --ir test_ir.jpg
"""

import torch
import numpy as np
import cv2
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class PalmVeinClassifier:
    """
    Palm vein classifier using trained few-shot model.
    """
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = None,
        device: str = 'cpu'
    ):
        self.device = device
        self.data_dir = data_dir
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model, self.config = self._load_model(model_path)
        self.model.eval()
        
        # Load metadata if available
        self.label_to_name = {}
        self.name_to_label = {}
        
        metadata_path = Path(model_path).parent / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.label_to_name = {int(k): v for k, v in metadata['label_to_name'].items()}
            self.name_to_label = metadata['name_to_label']
        
        logger.info(f"Model loaded. Embedding dim: {self.config.get('embedding_dim', 128)}")
    
    def _load_model(self, model_path: str):
        """Load trained model."""
        from model import PrototypicalNetwork
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        
        model = PrototypicalNetwork(
            embedding_dim=config.get('embedding_dim', 128),
            dropout=0.0,  # No dropout for inference
            backbone=config.get('backbone', 'custom'),
            pretrained=False,
            temperature=config.get('temperature', 0.5)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model, config
    
    def preprocess(self, image_path: str, is_ir: bool = False) -> torch.Tensor:
        """Load and preprocess image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load: {image_path}")
        
        image_size = self.config.get('image_size', (112, 112))
        img = cv2.resize(img, image_size)
        
        if is_ir and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        return torch.from_numpy(img).unsqueeze(0)
    
    def enroll_from_dataset(self, data_dir: str = None):
        """Enroll all persons from dataset."""
        from dataset import FewShotDataset
        
        data_dir = data_dir or self.data_dir
        if not data_dir:
            raise ValueError("No data_dir specified")
        
        logger.info(f"Enrolling from {data_dir}")
        
        dataset = FewShotDataset(
            data_dir=data_dir,
            image_size=self.config.get('image_size', (112, 112)),
            mode='val'
        )
        
        self.model.clear_prototypes()
        
        for class_idx in dataset.classes:
            sample = dataset.get_sample(class_idx, sample_idx=0)
            rgb = sample['rgb'].unsqueeze(0).to(self.device)
            ir = sample['ir'].unsqueeze(0).to(self.device)
            self.model.enroll(class_idx, sample['name'], rgb, ir)
        
        self.label_to_name = dataset.label_to_name
        self.name_to_label = dataset.name_to_label
        
        logger.info(f"Enrolled {len(self.model.prototypes)} persons")
    
    def enroll_person(self, name: str, rgb_path: str, ir_path: str):
        """Enroll a single person."""
        rgb = self.preprocess(rgb_path, is_ir=False).to(self.device)
        ir = self.preprocess(ir_path, is_ir=True).to(self.device)
        
        class_idx = len(self.model.prototypes)
        self.model.enroll(class_idx, name, rgb, ir)
        
        self.label_to_name[class_idx] = name
        self.name_to_label[name] = class_idx
        
        logger.info(f"Enrolled: {name} (class {class_idx})")
    
    def classify(
        self,
        rgb_path: str,
        ir_path: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Classify a person.
        
        Returns:
            List of (name, confidence) tuples
        """
        if not self.model.prototypes:
            raise ValueError("No persons enrolled! Call enroll_from_dataset first.")
        
        rgb = self.preprocess(rgb_path, is_ir=False).to(self.device)
        ir = self.preprocess(ir_path, is_ir=True).to(self.device)
        
        top_classes, top_scores = self.model.classify(rgb, ir, top_k=top_k)
        
        results = []
        for class_idx, score in zip(top_classes.tolist(), top_scores.tolist()):
            name = self.model.class_names.get(class_idx, f"Unknown_{class_idx}")
            results.append((name, score))
        
        return results
    
    def identify(
        self,
        rgb_path: str,
        ir_path: str,
        threshold: float = 0.5
    ) -> Tuple[str, float]:
        """
        Identify a person (returns best match or 'unknown').
        """
        results = self.classify(rgb_path, ir_path, top_k=1)
        
        if results and results[0][1] >= threshold:
            return results[0]
        else:
            return ("unknown", 0.0)
    
    def evaluate(self, data_dir: str = None, num_test: int = 200) -> Dict:
        """
        Evaluate classifier on dataset.
        """
        from dataset import FewShotDataset
        import random
        
        data_dir = data_dir or self.data_dir
        
        # Enroll first
        self.enroll_from_dataset(data_dir)
        
        # Load dataset for testing
        dataset = FewShotDataset(data_dir=data_dir, mode='val')
        
        # Test
        top1, top5, top10 = 0, 0, 0
        total = 0
        
        test_samples = []
        for class_idx in dataset.classes:
            samples = dataset.class_to_samples[class_idx]
            for idx in range(len(samples)):
                test_samples.append((class_idx, idx))
        
        random.shuffle(test_samples)
        test_samples = test_samples[:num_test]
        
        for class_idx, sample_idx in test_samples:
            sample = dataset.class_to_samples[class_idx][sample_idx]
            
            results = self.classify(sample['rgb_path'], sample['ir_path'], top_k=10)
            pred_names = [r[0] for r in results]
            true_name = self.label_to_name.get(class_idx, "")
            
            if pred_names and pred_names[0] == true_name:
                top1 += 1
            if true_name in pred_names[:5]:
                top5 += 1
            if true_name in pred_names[:10]:
                top10 += 1
            
            total += 1
        
        return {
            'top1_accuracy': 100 * top1 / total if total > 0 else 0,
            'top5_accuracy': 100 * top5 / total if total > 0 else 0,
            'top10_accuracy': 100 * top10 / total if total > 0 else 0,
            'total_tested': total
        }
    
    def save_database(self, path: str):
        """Save enrolled prototypes."""
        data = {
            'prototypes': {k: v.tolist() for k, v in self.model.prototypes.items()},
            'class_names': self.model.class_names,
            'label_to_name': {str(k): v for k, v in self.label_to_name.items()},
            'name_to_label': self.name_to_label
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Database saved to {path}")
    
    def load_database(self, path: str):
        """Load enrolled prototypes."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.model.prototypes = {
            int(k): torch.tensor(v) for k, v in data['prototypes'].items()
        }
        self.model.class_names = {int(k): v for k, v in data['class_names'].items()}
        self.label_to_name = {int(k): v for k, v in data['label_to_name'].items()}
        self.name_to_label = data['name_to_label']
        
        logger.info(f"Database loaded: {len(self.model.prototypes)} persons")


def main():
    parser = argparse.ArgumentParser(description="Palm Vein Classification")
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--data_dir', type=str, default='final_folder', help='Dataset directory')
    parser.add_argument('--rgb', type=str, help='RGB image path for single classification')
    parser.add_argument('--ir', type=str, help='IR image path for single classification')
    parser.add_argument('--evaluate', action='store_true', help='Run full evaluation')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = PalmVeinClassifier(
        model_path=args.model,
        data_dir=args.data_dir
    )
    
    if args.evaluate:
        # Full evaluation
        print("\n" + "=" * 50)
        print("EVALUATING ON DATASET")
        print("=" * 50)
        
        results = classifier.evaluate(args.data_dir, num_test=300)
        
        print(f"\nTop-1 Accuracy:  {results['top1_accuracy']:.2f}%")
        print(f"Top-5 Accuracy:  {results['top5_accuracy']:.2f}%")
        print(f"Top-10 Accuracy: {results['top10_accuracy']:.2f}%")
        print(f"Total tested: {results['total_tested']}")
        print("=" * 50)
    
    elif args.rgb and args.ir:
        # Single classification
        classifier.enroll_from_dataset()
        
        print("\n" + "=" * 50)
        print("CLASSIFICATION RESULTS")
        print("=" * 50)
        
        results = classifier.classify(args.rgb, args.ir, top_k=args.top_k)
        
        for i, (name, score) in enumerate(results, 1):
            print(f"  {i}. {name}: {score:.4f}")
        
        print("=" * 50)
    
    else:
        print("Usage:")
        print("  Evaluate: python inference.py --model path/model.pth --evaluate")
        print("  Classify: python inference.py --model path/model.pth --rgb img.jpg --ir ir.jpg")


if __name__ == "__main__":
    main()
