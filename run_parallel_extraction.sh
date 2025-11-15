#!/bin/bash

# Parallel Extraction Launcher
# Runs 3 Modal workers simultaneously to process 15 PGNs

echo "🚀 PARALLEL EXTRACTION LAUNCHER"
echo "================================"
echo "Launching 3 workers to process 15 PGNs in parallel..."
echo ""

# Kill any existing extraction jobs
pkill -f "extract_worker" 2>/dev/null

# Launch all 3 workers in background
echo "Starting Worker 1 (PGNs 1-5)..."
nohup modal run extract_worker_1.py > worker1.log 2>&1 &
WORKER1_PID=$!

echo "Starting Worker 2 (PGNs 6-10)..."
nohup modal run extract_worker_2.py > worker2.log 2>&1 &
WORKER2_PID=$!

echo "Starting Worker 3 (PGNs 11-15)..."
nohup modal run extract_worker_3.py > worker3.log 2>&1 &
WORKER3_PID=$!

echo ""
echo "✓ All 3 workers launched!"
echo "  Worker 1 PID: $WORKER1_PID (log: worker1.log)"
echo "  Worker 2 PID: $WORKER2_PID (log: worker2.log)"  
echo "  Worker 3 PID: $WORKER3_PID (log: worker3.log)"
echo ""
echo "Monitor progress:"
echo "  tail -f worker1.log"
echo "  tail -f worker2.log"
echo "  tail -f worker3.log"
echo ""
echo "Or check all at once:"
echo "  tail -f worker*.log"
echo ""
echo "ETA: ~40-60 minutes for all workers to complete"

