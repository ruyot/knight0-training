#!/bin/bash

echo "================================================================================================="
echo "🚀 KNIGHT0 KAGGLE TRAINING PIPELINE"
echo "================================================================================================="
echo ""
echo "This will:"
echo "  1. Download 109M position Kaggle dataset (9.79 GB)"
echo "  2. Process ALL 109M positions to training format (~10 GB)"
echo "  3. Upload to Modal volume"
echo "  4. Train LARGE model on A100 GPU (30 epochs, ~8-12 hours)"
echo "  5. Export to ONNX"
echo "  6. Download model for deployment"
echo ""
echo "FULL POWER MODE: 109M positions + Large model + A100 GPU"
echo ""
echo ""
echo ""
echo "================================================================================================="
echo ""

# Check if kagglehub is installed
if ! python3 -c "import kagglehub" 2>/dev/null; then
    echo "📦 Installing kagglehub..."
    pip3 install kagglehub
    echo ""
fi

# Step 1: Download & Process
echo "STEP 1: Download & Process Kaggle Dataset"
echo "─────────────────────────────────────────────────────────────────────────────────────────────────"
echo ""
echo "⚠️  NOTE: You may need to authenticate with Kaggle."
echo "   If prompted, follow the instructions to set up your API key."
echo ""
read -p "Press ENTER to start download..."
echo ""

python3 download_kaggle_dataset.py

if [ ! -f "kaggle_training_data.pkl" ]; then
    echo ""
    echo "❌ Download failed! Please check errors above."
    exit 1
fi

echo ""
echo "✅ Dataset ready!"
echo ""

# Step 2: Upload to Modal
echo "STEP 2: Upload to Modal Volume"
echo "─────────────────────────────────────────────────────────────────────────────────────────────────"
echo ""

modal volume put knight0-volume kaggle_training_data.pkl /kaggle_training_data.pkl

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Upload failed!"
    exit 1
fi

echo ""
echo "✅ Dataset uploaded to Modal!"
echo ""

# Step 3: Train
echo "STEP 3: Train on Modal GPU (A100 - FULL POWER!)"
echo "─────────────────────────────────────────────────────────────────────────────────────────────────"
echo ""
echo "🔥 Training LARGE model on ALL 109M positions!"
echo ""
echo "⏱️  This will take 8-12 hours. You can:"
echo "   - Let it run (recommended)"
echo "   - Close terminal (training continues on Modal)"
echo "   - Monitor at: https://modal.com/apps"
echo ""
read -p "Press ENTER to start training..."
echo ""

modal run train_on_kaggle.py 2>&1 | tee kaggle_training.log

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed! Check kaggle_training.log"
    exit 1
fi

echo ""
echo "✅ Training complete!"
echo ""

# Step 4: Download model
echo "STEP 4: Download Trained Model"
echo "─────────────────────────────────────────────────────────────────────────────────────────────────"
echo ""

modal volume get knight0-volume /knight0_model.onnx knight0_model.onnx

if [ ! -f "knight0_model.onnx" ]; then
    echo ""
    echo "❌ Download failed!"
    exit 1
fi

echo ""
echo "✅ Model downloaded!"
echo ""

# Done!
echo "================================================================================================="
echo "🎉 SUCCESS! KNIGHT0 MODEL TRAINED!"
echo "================================================================================================="
echo ""
echo "Model: knight0_model.onnx"
echo "Size: $(du -h knight0_model.onnx | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Test locally: python3 test_model.py"
echo "  2. Add search layer: python3 search_engine.py"
echo "  3. Deploy to ChessHacks inference repo"
echo "  4. Upload to Hugging Face (optional)"
echo ""
echo "Expected performance: 2000-2200+ Elo! (LARGE model + 109M positions)"
echo ""
echo "================================================================================================="

