"""
Download and process Kaggle dataset directly on Modal.
Avoids local disk space issues.
"""

import modal

app = modal.App("kaggle-downloader")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "kagglehub",
    "pandas",
)

volume = modal.Volume.from_name("knight0-volume", create_if_missing=True)
VOLUME_PATH = "/root/knight0"


@app.function(
    image=image,
    timeout=7200,  # 2 hours for download + processing
    volumes={VOLUME_PATH: volume},
    cpu=4,  # More CPU for faster processing
)
def download_and_process():
    """
    Download Kaggle dataset and process it into training format.
    All happens on Modal with plenty of disk space.
    """
    import kagglehub
    import pandas as pd
    import pickle
    from pathlib import Path
    
    print("="*80)
    print("DOWNLOADING KAGGLE CHESS DATASET (109M POSITIONS)")
    print("="*80)
    print("")
    
    # Download dataset
    print("Downloading from Kaggle (4.25 GB compressed)...")
    path = kagglehub.dataset_download("joannpeeler/labeled-chess-positions-109m-csv-format")
    print(f"Dataset downloaded to: {path}")
    print("")
    
    # Find CSV file
    dataset_path = Path(path)
    csv_files = list(dataset_path.glob("*.csv"))
    
    if not csv_files:
        print("Error: No CSV files found!")
        return
    
    csv_file = csv_files[0]
    print(f"Found dataset: {csv_file}")
    print(f"   Size: {csv_file.stat().st_size / (1024**3):.2f} GB")
    print("")
    
    # Process in chunks
    print("Processing dataset (converting to training format)...")
    print("   This will take 10-20 minutes...")
    print("")
    
    chunk_size = 100000
    output_chunks = []
    total_positions = 0
    max_positions = None  # Process all
    
    print(f"Target: ALL POSITIONS (~109M)")
    print("")
    
    for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
        if max_positions and total_positions >= max_positions:
            break
        
        # Extract relevant columns (no chess parsing - much faster!)
        positions = []
        
        for _, row in chunk.iterrows():
            if max_positions and total_positions >= max_positions:
                break
            
            try:
                fen = row['FEN']
                eval_score = int(row['Eval'])
                result = float(row['Result'])
                
                # Normalize value (-1 to 1 range)
                value = max(-1.0, min(1.0, eval_score / 1000.0))
                
                # Store position with eval - we'll handle moves during training
                positions.append({
                    "board": fen,
                    "move": None,  # Will be determined from FEN during training
                    "value": value
                })
                
                total_positions += 1
                
            except Exception:
                continue
        
        output_chunks.append(positions)
        
        # Progress update
        if (chunk_num + 1) % 10 == 0:
            print(f"  Processed {total_positions:,} positions...")
    
    print("")
    print(f"Processed {total_positions:,} positions")
    print("")
    
    # Combine and save
    print("Saving to pickle format...")
    all_positions = []
    for chunk in output_chunks:
        all_positions.extend(chunk)
    
    output_file = Path(VOLUME_PATH) / "kaggle_training_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_positions, f)
    
    volume.commit()
    
    print(f"Saved to: {output_file}")
    print(f"  Size: {output_file.stat().st_size / (1024**2):.1f} MB")
    print(f"  Positions: {len(all_positions):,}")
    print("")
    
    print("="*80)
    print("KAGGLE DATASET READY FOR TRAINING")
    print("="*80)
    print("")
    print("Next step:")
    print("  modal run train_on_kaggle.py")
    print("")
    
    return {
        "total_positions": len(all_positions),
        "output_path": str(output_file)
    }


@app.local_entrypoint()
def main():
    """Run the download and processing on Modal"""
    print("Starting download and processing on Modal...")
    print("")
    result = download_and_process.remote()
    print("")
    print(f"Complete! Processed {result['total_positions']:,} positions")
    print(f"Data saved to Modal volume: {result['output_path']}")

