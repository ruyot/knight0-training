"""
Manage training data (clean, rebuild, inspect).
"""

import modal
from pathlib import Path
import shutil

# Import the modal app
from modal_train import app, volume, VOLUME_PATH, image


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def clean_shards():
    """Delete all shard files (forces reprocessing)."""
    from pathlib import Path
    
    shards_dir = Path(VOLUME_PATH) / "processed"
    
    if shards_dir.exists():
        count = 0
        for shard in shards_dir.glob("*.pkl"):
            shard.unlink()
            count += 1
            print(f"Deleted: {shard.name}")
        
        volume.commit()
        print(f"\n✓ Deleted {count} shard files")
        print("Next training run will reprocess all PGNs")
    else:
        print("No shards directory found")


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def inspect_volume():
    """List all files in the volume with sizes."""
    from pathlib import Path
    
    root = Path(VOLUME_PATH)
    
    if not root.exists():
        print("Volume is empty")
        return
    
    print("\n" + "="*80)
    print(f"Volume Contents: {VOLUME_PATH}")
    print("="*80)
    
    # Group by type
    shards = []
    checkpoints = []
    data_files = []
    other = []
    
    for item in sorted(root.rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / 1e6
            rel_path = str(item.relative_to(root))
            
            if 'shard_' in rel_path:
                shards.append((rel_path, size_mb))
            elif 'checkpoint' in rel_path or 'model.pth' in rel_path:
                checkpoints.append((rel_path, size_mb))
            elif rel_path.endswith('.pkl') or rel_path.endswith('.onnx'):
                data_files.append((rel_path, size_mb))
            else:
                other.append((rel_path, size_mb))
    
    if shards:
        print("\n📦 Shards:")
        for path, size in shards:
            print(f"  {path:<60} {size:>8.2f} MB")
        print(f"  Total: {sum(s for _, s in shards):.2f} MB")
    
    if data_files:
        print("\n📊 Data Files:")
        for path, size in data_files:
            print(f"  {path:<60} {size:>8.2f} MB")
        print(f"  Total: {sum(s for _, s in data_files):.2f} MB")
    
    if checkpoints:
        print("\n💾 Checkpoints:")
        for path, size in checkpoints:
            print(f"  {path:<60} {size:>8.2f} MB")
        print(f"  Total: {sum(s for _, s in checkpoints):.2f} MB")
    
    if other:
        print("\n📄 Other:")
        for path, size in other:
            print(f"  {path:<60} {size:>8.2f} MB")
    
    total_size = sum(s for _, s in shards + checkpoints + data_files + other)
    print(f"\n{'Total Volume Size:':<62} {total_size:>8.2f} MB")
    print("="*80)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def delete_training_data():
    """Delete training_data.pkl (forces regeneration from shards or PGNs)."""
    from pathlib import Path
    
    data_file = Path(VOLUME_PATH) / "training_data.pkl"
    
    if data_file.exists():
        size_mb = data_file.stat().st_size / 1e6
        data_file.unlink()
        volume.commit()
        print(f"✓ Deleted training_data.pkl ({size_mb:.2f} MB)")
        print("Shards are preserved - next run will merge them")
    else:
        print("training_data.pkl not found")


@app.local_entrypoint()
def manage(action: str = "inspect"):
    """
    Manage training data.
    
    Args:
        action: Action to perform
            - "inspect": List all files in volume
            - "clean-shards": Delete all shards (reprocess everything)
            - "clean-data": Delete training_data.pkl (regenerate from shards)
            - "clean-all": Clear entire volume
    """
    if action == "inspect":
        print("📋 Inspecting volume...")
        inspect_volume.remote()
    
    elif action == "clean-shards":
        response = input("⚠ Delete all shards? This will force reprocessing. (yes/no): ")
        if response.lower() == "yes":
            clean_shards.remote()
        else:
            print("Cancelled")
    
    elif action == "clean-data":
        response = input("⚠ Delete training_data.pkl? (yes/no): ")
        if response.lower() == "yes":
            delete_training_data.remote()
        else:
            print("Cancelled")
    
    elif action == "clean-all":
        print("⚠⚠⚠ WARNING: This will delete EVERYTHING in the volume!")
        print("This includes:")
        print("  - All shards")
        print("  - All checkpoints")
        print("  - All training data")
        print("  - The trained ONNX model")
        response = input("\nType 'DELETE EVERYTHING' to confirm: ")
        if response == "DELETE EVERYTHING":
            from modal_train import clear_volume
            clear_volume.remote()
        else:
            print("Cancelled")
    
    else:
        print(f"Unknown action: {action}")
        print("Available actions: inspect, clean-shards, clean-data, clean-all")


if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    manage(action=action)

