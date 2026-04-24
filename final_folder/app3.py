import os
import re
import shutil
from pathlib import Path

# ============================================
# CONFIGURE THESE PATHS
# ============================================
SOURCE_DIR = r"Palm dataset"      # New dataset with ir/ and rgb/ folders
OUTPUT_DIR = r"final_folder"      # Existing folder with person_0001 to person_0453
START_FROM = 454                  # Continue from person_0454
# ============================================


def extract_identifier(filename):
    """
    Extract the unique identifier from filename.
    
    Examples:
    'ir (1).jpg' -> '1'
    'ir (1)a.jpg' -> '1a'
    'ir (1)aa.jpg' -> '1aa'
    'ir (1)aaa.jpg' -> '1aaa'
    'ir (2) 21.jpg' -> '2_21'
    'ir (2) 31.jpg' -> '2_31'
    'rgb (1)1.jpg' -> '1_1'
    'ir ()1.jpg' -> '0_1'
    'ir (0)1.jpg' -> '0_1'
    """
    # Remove extension
    name = Path(filename).stem.lower()
    
    # Remove 'ir' or 'rgb' prefix
    name = re.sub(r'^(ir|rgb)\s*', '', name)
    
    # Pattern: (number) followed by optional suffix
    # Examples: (1), (1)a, (1)aa, (2) 21, (1)1
    
    # Match pattern like "(1)a" or "(1) 21" or "(1)aaa"
    match = re.match(r'\((\d*)\)\s*(\S*)', name)
    
    if match:
        num = match.group(1) if match.group(1) else '0'
        suffix = match.group(2).strip() if match.group(2) else ''
        
        # Clean suffix - remove spaces, combine
        suffix = re.sub(r'\s+', '_', suffix)
        
        if suffix:
            return f"{num}_{suffix}"
        else:
            return num
    
    return None


def organize_palm_dataset(source_dir, output_dir, start_from):
    """Organize Palm dataset into person folders."""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    ir_folder = source_path / "ir"
    rgb_folder = source_path / "rgb"
    
    if not ir_folder.exists():
        print(f"ERROR: IR folder not found: {ir_folder}")
        return
    
    if not rgb_folder.exists():
        print(f"ERROR: RGB folder not found: {rgb_folder}")
        return
    
    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    ir_images = {}   # identifier -> filepath
    rgb_images = {}  # identifier -> filepath
    
    # Process IR images
    print("Processing IR folder...")
    for img_file in ir_folder.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
            identifier = extract_identifier(img_file.name)
            if identifier:
                ir_images[identifier] = img_file
                # Debug: print first few
                if len(ir_images) <= 5:
                    print(f"  IR: '{img_file.name}' -> ID: '{identifier}'")
    
    print(f"Found {len(ir_images)} IR images")
    
    # Process RGB images
    print("\nProcessing RGB folder...")
    for img_file in rgb_folder.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
            identifier = extract_identifier(img_file.name)
            if identifier:
                rgb_images[identifier] = img_file
                # Debug: print first few
                if len(rgb_images) <= 5:
                    print(f"  RGB: '{img_file.name}' -> ID: '{identifier}'")
    
    print(f"Found {len(rgb_images)} RGB images")
    
    # Find matching pairs
    common_ids = sorted(set(ir_images.keys()) & set(rgb_images.keys()))
    print(f"\nMatched pairs: {len(common_ids)}")
    
    # Show unmatched for debugging
    ir_only = set(ir_images.keys()) - set(rgb_images.keys())
    rgb_only = set(rgb_images.keys()) - set(ir_images.keys())
    
    if ir_only:
        print(f"\nIR images without RGB match ({len(ir_only)}):")
        for id in list(ir_only)[:10]:
            print(f"  - {id}: {ir_images[id].name}")
        if len(ir_only) > 10:
            print(f"  ... and {len(ir_only) - 10} more")
    
    if rgb_only:
        print(f"\nRGB images without IR match ({len(rgb_only)}):")
        for id in list(rgb_only)[:10]:
            print(f"  - {id}: {rgb_images[id].name}")
        if len(rgb_only) > 10:
            print(f"  ... and {len(rgb_only) - 10} more")
    
    # Create person folders
    print(f"\nCreating person folders starting from person_{start_from:04d}...")
    
    person_counter = start_from
    created_count = 0
    
    for identifier in common_ids:
        person_id = f"person_{person_counter:04d}"
        person_dir = output_path / person_id
        person_dir.mkdir(exist_ok=True)
        
        ir_source = ir_images[identifier]
        rgb_source = rgb_images[identifier]
        
        # Copy files
        shutil.copy2(ir_source, person_dir / f"ir{ir_source.suffix.lower()}")
        shutil.copy2(rgb_source, person_dir / f"rgb{rgb_source.suffix.lower()}")
        
        if created_count < 10:
            print(f"  Created: {person_id} <- ir:'{ir_source.name}' + rgb:'{rgb_source.name}'")
        
        person_counter += 1
        created_count += 1
    
    if created_count > 10:
        print(f"  ... and {created_count - 10} more")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"COMPLETED!")
    print(f"{'='*60}")
    print(f"New persons created: {created_count}")
    print(f"Person range: person_{start_from:04d} to person_{person_counter-1:04d}")
    print(f"Total persons in {output_dir}: {person_counter - 1}")
    print(f"{'='*60}")


if __name__ == '__main__':
    print("="*60)
    print("PALM DATASET ORGANIZER (IR/RGB Folder Structure)")
    print("="*60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Starting from: person_{START_FROM:04d}")
    print("="*60 + "\n")
    
    organize_palm_dataset(SOURCE_DIR, OUTPUT_DIR, START_FROM)