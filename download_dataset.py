import os
import requests
import zipfile
import shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# URLs to download
urls = [
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Test_Input.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Test_GroundTruth.zip"
]

def download_file(url, dest_folder="data"):
    """Download a file with progress bar"""
    # Create destination folder if it doesn't exist
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_folder, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        print(f"✓ {filename} already exists, skipping download...")
        return filepath
    
    print(f"Downloading {filename}...")
    
    try:
        # Stream the download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"✓ {filename} downloaded successfully\n")
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {filename}: {e}\n")
        return None

def unzip_file(zip_path, dest_folder="data"):
    """Unzip a file and show progress"""
    if not os.path.exists(zip_path):
        print(f"✗ {zip_path} not found, skipping unzip")
        return
    
    filename = os.path.basename(zip_path)
    print(f"Unzipping {filename}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in archive
            file_list = zip_ref.namelist()
            
            # Extract with progress bar
            for file in tqdm(file_list, desc=f"Extracting {filename}"):
                zip_ref.extract(file, dest_folder)
        
        print(f"✓ {filename} extracted successfully\n")
        return True
        
    except zipfile.BadZipFile as e:
        print(f"✗ Error unzipping {filename}: {e}\n")
        return False

def organize_dataset(base_folder="data"):
    """Organize the dataset into raw folder structure"""
    print("\n" + "=" * 50)
    print("Organizing dataset into structured folders...\n")
    
    # Define the mapping of source to destination
    mappings = [
        ("ISIC2018_Task1-2_Training_Input", "raw/isic_2018_train/images"),
        ("ISIC2018_Task1_Training_GroundTruth", "raw/isic_2018_train/masks"),
        ("ISIC2018_Task1-2_Test_Input", "raw/isic_2018_test/images"),
        ("ISIC2018_Task1_Test_GroundTruth", "raw/isic_2018_test/masks"),
        ("ISIC2018_Task1-2_Validation_Input", "raw/isic_2018_val/images"),
        ("ISIC2018_Task1_Validation_GroundTruth", "raw/isic_2018_val/masks"),
    ]
    
    # Create the raw folder structure
    raw_folders = [
        "raw/isic_2018_train/images",
        "raw/isic_2018_train/masks",
        "raw/isic_2018_test/images",
        "raw/isic_2018_test/masks",
        "raw/isic_2018_val/images",
        "raw/isic_2018_val/masks",
    ]
    
    for folder in raw_folders:
        Path(os.path.join(base_folder, folder)).mkdir(parents=True, exist_ok=True)
    
    print("✓ Created folder structure\n")
    
    # Move files according to mappings
    for source_folder, dest_folder in mappings:
        source_path = os.path.join(base_folder, source_folder)
        dest_path = os.path.join(base_folder, dest_folder)
        
        if not os.path.exists(source_path):
            print(f"⚠ Warning: {source_folder} not found, skipping...")
            continue
        
        # Get all jpg and png files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(source_path).rglob(ext))
        
        if not image_files:
            print(f"⚠ No image files found in {source_folder}")
            continue
        
        print(f"Moving {len(image_files)} files from {source_folder} to {dest_folder}")
        
        # Move files with progress bar
        for img_file in tqdm(image_files, desc=f"Moving to {dest_folder}"):
            dest_file = os.path.join(dest_path, img_file.name)
            shutil.move(str(img_file), dest_file)
        
        print(f"✓ Completed moving files to {dest_folder}\n")
    
    # Clean up empty source folders
    print("Cleaning up empty source folders...")
    for source_folder, _ in mappings:
        source_path = os.path.join(base_folder, source_folder)
        if os.path.exists(source_path):
            try:
                shutil.rmtree(source_path)
                print(f"✓ Removed {source_folder}")
            except Exception as e:
                print(f"⚠ Could not remove {source_folder}: {e}")
    
    print("\n✓ Dataset organization complete!")

def preprocess_image(image_path, target_size=(256, 256), normalize=True):
    """Preprocess a single image"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] if requested
    if normalize:
        img = img.astype(np.float32) / 255.0
    
    return img

def preprocess_mask(mask_path, target_size=(256, 256)):
    """Preprocess a single mask"""
    # Read mask (grayscale)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    
    # Resize using nearest neighbor to preserve binary values
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Binarize (threshold at 127)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Convert to 0 and 1
    mask = (mask / 255).astype(np.uint8)
    
    return mask

def preprocess_dataset(base_folder="data", target_size=(256, 256)):
    """
    Preprocess the entire dataset and save as .npy files
    
    Args:
        base_folder: Base directory containing the raw data
        target_size: Target size (height, width)
    """
    print("\n" + "=" * 50)
    print("PREPROCESSING DATASET")
    print("=" * 50)
    
    raw_dir = Path(base_folder) / "raw"
    processed_dir = Path(base_folder) / "processed"
    
    datasets = [
        ('isic_2018_train', 'train'),
        ('isic_2018_val', 'val'),
        ('isic_2018_test', 'test')
    ]
    
    for raw_name, proc_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {proc_name}")
        print(f"{'='*50}")
        
        # Create output directories
        dataset_output = processed_dir / proc_name
        (dataset_output / "images").mkdir(parents=True, exist_ok=True)
        (dataset_output / "masks").mkdir(parents=True, exist_ok=True)
        
        # Input paths
        input_images = raw_dir / raw_name / "images"
        input_masks = raw_dir / raw_name / "masks"
        
        # Get list of images
        image_files = list(input_images.glob("*.jpg")) + list(input_images.glob("*.png"))
        
        if not image_files:
            print(f"⚠ No images found in {input_images}")
            continue
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        processed_count = 0
        for img_path in tqdm(image_files, desc=f"Processing {proc_name}"):
            img_id = img_path.stem
            
            # Determine mask path
            mask_path = None
            if input_masks:
                mask_path = input_masks / f"{img_id}_segmentation.png"
                if not mask_path.exists():
                    # Try without _segmentation suffix
                    mask_path = input_masks / f"{img_id}.png"
                    if not mask_path.exists():
                        print(f"\n⚠ Mask not found for {img_id}, skipping...")
                        continue
            
            try:
                # Process image
                image_proc = preprocess_image(img_path, target_size, normalize=True)
                
                # Save processed image
                img_output = dataset_output / "images" / f"{img_id}.npy"
                np.save(img_output, image_proc)
                
                # Process and save mask if exists
                if mask_path:
                    mask_proc = preprocess_mask(mask_path, target_size)
                    mask_output = dataset_output / "masks" / f"{img_id}.npy"
                    np.save(mask_output, mask_proc)
                
                processed_count += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {img_id}: {e}")
                continue
        
        print(f"✓ Successfully processed {processed_count}/{len(image_files)} images for {proc_name}")
    
    print("\n✅ Preprocessing completed!")
    
    # Show final statistics
    print("\n" + "="*50)
    print("FINAL PREPROCESSING STATISTICS")
    print("="*50)
    
    for _, proc_name in datasets:
        dataset_path = processed_dir / proc_name / "images"
        n_images = len(list(dataset_path.glob("*.npy"))) if dataset_path.exists() else 0
        
        masks_path = processed_dir / proc_name / "masks"
        n_masks = len(list(masks_path.glob("*.npy"))) if masks_path.exists() else 0
        print(f"{proc_name.upper()}: {n_images} images, {n_masks} masks")

def main():
    print("ISIC 2018 Challenge Dataset Downloader & Preprocessor")
    print("=" * 50)
    print(f"Downloading {len(urls)} files...\n")
    
    downloaded_files = []
    
    # Download all files
    for url in urls:
        filepath = download_file(url)
        if filepath:
            downloaded_files.append(filepath)
    
    print("\n" + "=" * 50)
    print("Unzipping files...\n")
    
    # Unzip all downloaded files
    for filepath in downloaded_files:
        unzip_file(filepath)
    
    print("=" * 50)
    print("Deleting original zip files...\n")
    
    # Delete zip files
    for filepath in downloaded_files:
        try:
            os.remove(filepath)
            print(f"✓ Deleted {os.path.basename(filepath)}")
        except Exception as e:
            print(f"✗ Error deleting {os.path.basename(filepath)}: {e}")
    
    # Organize the dataset
    organize_dataset()
    
    # Preprocess the dataset
    preprocess_dataset()
    
    print("\n" + "=" * 50)
    print("COMPLETE! Dataset is ready for training")
    print("\nFinal structure:")
    print("  data/")
    print("    ├── raw/              (original organized data)")
    print("    │   ├── isic_2018_train/")
    print("    │   ├── isic_2018_test/")
    print("    │   └── isic_2018_val/")
    print("    └── processed/        (preprocessed .npy files)")
    print("        ├── train/")
    print("        │   ├── images/")
    print("        │   └── masks/")
    print("        ├── test/")
    print("        │   └── images/")
    print("        │   └── masks/")
    print("        └── val/")
    print("            ├── images/")
    print("            └── masks/")

if __name__ == "__main__":
    main()