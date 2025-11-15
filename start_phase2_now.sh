#!/bin/bash

# Start Phase 2 extraction IN PARALLEL with Phase 1
# Uses existing high-quality PGNs (Lichess Elite + TCEC)

echo "🚀 STARTING PHASE 2 EXTRACTION"
echo "=============================="
echo ""
echo "Phase 1 (depth 10) is still running..."
echo "Phase 2 (depth 25) will run in parallel!"
echo ""
echo "Using:"
echo "  - Lichess Elite 2023-12"
echo "  - 4 TCEC competition PGNs"
echo "  - Stockfish depth 25"
echo "  - MultiPV 3"
echo "  - Position filtering for tactical moments"
echo ""

# Launch Phase 2 extraction
nohup modal run phase2_extractor.py \
    --data-source lichess \
    --depth 25 \
    --multipv 3 \
    --max-chunks 5 \
    > phase2_extraction.log 2>&1 &

PHASE2_PID=$!

echo "✓ Phase 2 extraction launched!"
echo "  PID: $PHASE2_PID"
echo "  Log: phase2_extraction.log"
echo ""
echo "Monitor progress:"
echo "  tail -f phase2_extraction.log"
echo ""
echo "ETA: ~1-2 hours for 5 high-quality PGNs at depth 25"
echo ""
echo "This will create ~200K-500K HIGH-QUALITY positions"
echo "for Phase 2 fine-tuning!"

