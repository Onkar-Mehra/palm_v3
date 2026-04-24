"""
Quick Demo: How to use the Palm Biometric System
=================================================
Run this AFTER training to verify everything works.

This script:
1. Loads the trained model
2. Enrolls 3 test people from your dataset
3. Tries to identify them from their second image
4. Tries to identify with a "wrong" image to see rejection
"""

import sys
import os
from pathlib import Path

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    print("=" * 70)
    print("PALM BIOMETRIC SYSTEM - QUICK DEMO")
    print("=" * 70)
    
    # ===========================================================
    # STEP 1: Initialize
    # ===========================================================
    print("\n[1/4] Loading the trained model...")
    
    from identify import PalmBiometricSystem
    
    model_path = "models/best_model.pth"
    if not Path(model_path).exists():
        print(f"\nERROR: Model not found at {model_path}")
        print("You need to train the model first:")
        print("   python train.py")
        sys.exit(1)
    
    system = PalmBiometricSystem(
        model_path=model_path,
        database_path="enrolled_database/demo_database.pkl",
        device="cpu",
    )
    
    # Clear any previous demo enrollments
    for n in list(system.list_enrolled()):
        system.remove_enrolled(n)
    
    print(f"   Loaded. Match threshold: {system.match_threshold:.4f}")
    
    # ===========================================================
    # STEP 2: Enroll a few test people from the dataset
    # ===========================================================
    print("\n[2/4] Enrolling 3 test people from the ROI dataset...")
    
    data_dir = Path("final_folder_roi")
    if not data_dir.exists():
        print(f"\nERROR: ROI dataset not found at {data_dir}")
        print("Run palm_roi.py first to create it.")
        sys.exit(1)
    
    person_dirs = sorted([d for d in data_dir.iterdir()
                         if d.is_dir() and not d.name.startswith('_')])
    
    if len(person_dirs) < 3:
        print(f"Need at least 3 people in dataset, found {len(person_dirs)}")
        sys.exit(1)
    
    test_people = person_dirs[:3]
    enrolled = []
    
    for person_dir in test_people:
        name = f"Demo_{person_dir.name}"
        rgb = person_dir / "rgb.jpg"
        ir = person_dir / "ir.jpg"
        
        if not rgb.exists() or not ir.exists():
            print(f"   Skipping {name}: files not found")
            continue
        
        success = system.register(name, str(rgb), str(ir), verbose=False)
        if success:
            enrolled.append((name, person_dir))
            print(f"   ✓ Enrolled: {name}")
    
    print(f"\n   Total enrolled: {len(enrolled)}")
    
    # ===========================================================
    # STEP 3: Identify each person from their OWN image
    # ===========================================================
    print("\n[3/4] Identifying each enrolled person from their own image...")
    print("   (This is the easy case - same image used for enrollment AND query)")
    print()
    
    correct = 0
    for true_name, person_dir in enrolled:
        rgb = person_dir / "rgb.jpg"
        ir = person_dir / "ir.jpg"
        
        result = system.identify(str(rgb), str(ir), top_k=3)
        
        is_correct = (result.name == true_name and result.status == 'accepted')
        marker = "✓" if is_correct else "✗"
        if is_correct:
            correct += 1
        
        print(f"   {marker} True: {true_name}")
        print(f"      Predicted: {result.name} (conf: {result.confidence:.4f}, status: {result.status})")
        print()
    
    print(f"   Self-identification: {correct}/{len(enrolled)} correct")
    
    # ===========================================================
    # STEP 4: Try to identify someone NOT enrolled
    # ===========================================================
    print("\n[4/4] Trying to identify a person who is NOT enrolled...")
    print("   (Should be REJECTED with low confidence)")
    print()
    
    # Use a person we didn't enroll
    if len(person_dirs) > 3:
        unknown_dir = person_dirs[10]  # someone different
        rgb = unknown_dir / "rgb.jpg"
        ir = unknown_dir / "ir.jpg"
        
        result = system.identify(str(rgb), str(ir), top_k=3)
        
        print(f"   Query: {unknown_dir.name} (NOT in database)")
        print(f"   Predicted: {result.name}")
        print(f"   Confidence: {result.confidence:.4f}")
        print(f"   Status: {result.status}")
        
        if result.status == 'rejected_low_confidence':
            print(f"\n   ✓ Correctly rejected as unknown")
        else:
            print(f"\n   ✗ Incorrectly accepted (false match)")
            print(f"     This may happen if threshold is too low.")
            print(f"     Current threshold: {system.match_threshold:.4f}")
    
    # ===========================================================
    # Summary
    # ===========================================================
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run real evaluation: python train.py (look at final eval_metrics)")
    print("  2. Enroll your real users:")
    print("     python identify.py register --name 'NAME' --rgb r.jpg --ir i.jpg")
    print("  3. Identify in production:")
    print("     python identify.py identify --rgb query_rgb.jpg --ir query_ir.jpg")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nDemo failed: {e}")
        print(traceback.format_exc())