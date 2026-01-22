import os
import shutil
from pathlib import Path
import random

# Configuration
DATASET_PATH = 'training/10-monkey-species/training'
SAMPLES_OUTPUT = 'static/samples'

# Species name mapping (folder name -> readable name)
SPECIES_MAPPING = {
    'n0': 'n0',
    'n1': 'n1',
    'n2': 'n2',
    'n3': 'n3',
    'n4': 'n4',
    'n5': 'n5',
    'n6': 'n6',
    'n7': 'n7',
    'n8': 'n8',
    'n9': 'n9'
}

def prepare_sample_images():
    """Extract one representative image from each species folder"""
    
    print("="*60)
    print("PREPARING SAMPLE IMAGES")
    print("="*60)
    
    # Create output directory
    os.makedirs(SAMPLES_OUTPUT, exist_ok=True)
    print(f"\nOutput directory: {SAMPLES_OUTPUT}")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\nError: Dataset not found at {DATASET_PATH}")
        print("Please download the dataset from Kaggle and extract it to the training folder.")
        return
    
    copied_count = 0
    
    # Process each species folder
    for folder_name, species_name in SPECIES_MAPPING.items():
        folder_path = os.path.join(DATASET_PATH, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"\nWarning: Folder not found: {folder_path}")
            continue
        
        # Get all image files in the folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        images = []
        
        for ext in image_extensions:
            images.extend(list(Path(folder_path).glob(f'*{ext}')))
            images.extend(list(Path(folder_path).glob(f'*{ext.upper()}')))
        
        if not images:
            print(f"\nWarning: No images found in {folder_name}")
            continue
        
        # Select a random image (or the first one)
        # Using middle image for consistency
        selected_image = images[len(images) // 2]
        
        # Determine output filename
        output_filename = f"{species_name}.jpg"
        output_path = os.path.join(SAMPLES_OUTPUT, output_filename)
        
        # Copy the image
        try:
            shutil.copy2(selected_image, output_path)
            print(f"✓ {species_name:30} -> {output_filename}")
            copied_count += 1
        except Exception as e:
            print(f"✗ Error copying {species_name}: {str(e)}")
    
    print("\n" + "="*60)
    print(f"COMPLETE: {copied_count}/{len(SPECIES_MAPPING)} sample images prepared")
    print("="*60)
    
    # List the created files
    print("\nSample images created:")
    for filename in sorted(os.listdir(SAMPLES_OUTPUT)):
        filepath = os.path.join(SAMPLES_OUTPUT, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  - {filename:30} ({size_kb:.1f} KB)")

if __name__ == '__main__':
    prepare_sample_images()