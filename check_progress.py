"""Check if shards are being created on Modal volume."""

import modal

app = modal.App("check-progress")
volume = modal.Volume.from_name("knight0-volume", create_if_missing=False)

@app.function(volumes={"/root/knight0": volume})
def check_volume():
    """Check what's on the volume."""
    import os
    from pathlib import Path
    
    processed_dir = Path("/root/knight0/processed")
    
    print("=" * 80)
    print("CHECKING MODAL VOLUME FOR PROGRESS")
    print("=" * 80)
    
    if not processed_dir.exists():
        print("❌ No 'processed' directory yet - extraction hasn't started saving")
        return
    
    shards = list(processed_dir.glob("shard_*.pkl"))
    
    if not shards:
        print("⏳ 'processed' directory exists but no shards yet")
        print("   Workers are probably still processing first PGN (< 50 games)")
        return
    
    print(f"✅ Found {len(shards)} shards!")
    print("")
    for shard in sorted(shards):
        size = shard.stat().st_size / (1024 * 1024)  # MB
        print(f"  {shard.name}: {size:.1f} MB")
    
    print("=" * 80)

@app.local_entrypoint()
def main():
    check_volume.remote()

