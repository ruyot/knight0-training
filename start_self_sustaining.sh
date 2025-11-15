#!/bin/bash

echo "🚀 STARTING SELF-SUSTAINING NEURAL NETWORK TRAINING"
echo "="
echo ""
echo "Step 1: Fast behavior cloning (NO Stockfish needed!)"
echo "  Extracting winning moves from PGNs..."
python3 behavior_clone.py

if [ ! -f "behavior_clone_data.pkl" ]; then
    echo "❌ Behavior cloning failed!"
    exit 1
fi

echo ""
echo "Step 2: Uploading behavior cloning data to Modal..."
modal volume put knight0-volume behavior_clone_data.pkl /behavior_clone_data.pkl

echo ""
echo "Step 3: Launching self-sustaining training loop on Modal GPU..."
echo "  This will run WHILE extraction continues!"
echo ""
modal run self_sustaining_loop.py

echo ""
echo "✓ Self-sustaining loop complete!"

