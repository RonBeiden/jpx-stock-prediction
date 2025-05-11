import os
import argparse
from pathlib import Path
import subprocess
import kaggle

def download_competition_data(
    competition_name: str = "jpx-tokyo-stock-exchange-prediction",
    output_dir: str = "data"
):
    """
    Download competition data using Kaggle API.
    
    Args:
        competition_name: Name of the Kaggle competition
        output_dir: Directory to save the data
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading data for competition: {competition_name}")
    
    # Download data using Kaggle API
    try:
        subprocess.run([
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition_name,
            "-p",
            str(output_path)
        ], check=True)
        print(f"\nData downloaded successfully to {output_path}")
        
    except subprocess.CalledProcessError as e:
        print("\nError downloading data. Make sure you have:")
        print("1. Installed the Kaggle API: pip install kaggle")
        print("2. Configured your Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("3. Accepted the competition rules on the Kaggle website")
        raise e

def download_jpx_data():
    """Download JPX data from Kaggle."""
    # Create data directories
    Path('data/raw/train').mkdir(parents=True, exist_ok=True)
    Path('data/raw/test').mkdir(parents=True, exist_ok=True)
    
    # Download data
    kaggle.api.competition_download_files(
        'jpx-tokyo-stock-exchange-prediction',
        path='data/raw'
    )

def main():
    parser = argparse.ArgumentParser(description="Download competition data")
    parser.add_argument("--output_dir", default="data",
                      help="Directory to save the data")
    
    args = parser.parse_args()
    download_competition_data(output_dir=args.output_dir)

if __name__ == "__main__":
    download_jpx_data()
    main() 