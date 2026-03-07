#!/bin/bash
# Monitor QHformer training progress

LOG_FILE="/Users/jiaoyuan/Desktop/QHformer/training_output.log"

echo "=========================================="
echo "QHformer Training Monitor"
echo "=========================================="
echo ""

# Check if training is running
if pgrep -f "train_qhformer.py" > /dev/null; then
    echo "✓ Training is running (PID: $(pgrep -f 'train_qhformer.py'))"
else
    echo "✗ Training is NOT running"
fi
echo ""

# Show latest log entries
echo "Latest training progress:"
echo "------------------------------------------"
tail -20 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
echo ""

# Show best MAE
echo "Best MAE achieved:"
echo "------------------------------------------"
grep "Best:" "$LOG_FILE" 2>/dev/null | tail -5 || echo "No MAE data yet"
echo ""

# Check for errors
echo "Error check:"
echo "------------------------------------------"
if grep -q "Error\|Exception\|Traceback" "$LOG_FILE" 2>/dev/null; then
    echo "⚠ Errors found in log:"
    grep "Error\|Exception\|Traceback" "$LOG_FILE" | tail -5
else
    echo "✓ No errors detected"
fi
echo ""

# Show training curves
echo "Training curves saved at:"
echo "------------------------------------------"
find /Users/jiaoyuan/Desktop/QHformer/training/runs -name "training_curves.png" -exec ls -la {} \; 2>/dev/null | tail -1
echo ""
