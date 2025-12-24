"""
Download Dataset Script

Simple script to download the Fruit Freshness dataset from Kaggle
and copy it to the project's data folder.

Usage:
    python utils/download_data.py
"""

import os
import sys
import shutil
from pathlib import Path


def main():
    # Get project paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    target_dir = data_dir / "dataset"
    
    # Check if already downloaded
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"✅ Dataset already exists at {target_dir}")
        return str(target_dir)
    
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system("pip install kagglehub -q")
        import kagglehub
    
    print("Downloading Fruit Freshness dataset from Kaggle...")
    print("This may take a few minutes (~3.5 GB)...")
    
    try:
        # Download using kagglehub (no API key required)
        cache_path = kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification")
        cache_path = Path(cache_path)
        
        print(f"Downloaded to cache: {cache_path}")
        
        # Copy to project data folder
        print(f"Copying to project: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy contents
        for item in cache_path.iterdir():
            dest = data_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        
        print(f"\n✅ Dataset ready at {data_dir}")
        
        # Show structure
        print("\nDataset structure:")
        for item in sorted(data_dir.rglob("*")):
            if item.is_dir():
                depth = len(item.relative_to(data_dir).parts)
                if depth <= 2:
                    num_files = sum(1 for _ in item.glob("*") if _.is_file())
                    print(f"  {'  ' * (depth-1)}📁 {item.name}/ ({num_files} items)")
        
        return str(data_dir)
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
