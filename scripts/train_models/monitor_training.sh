#!/bin/bash
# Monitor Read Correction Model Training Progress

echo "========================================"
echo "Read Correction Model Training Status"
echo "========================================"
echo ""

# Check running processes
echo "Running Processes:"
ps aux | grep -E "train_kmer|train_base_error" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    time=$(echo $line | awk '{print $10}')
    cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
    
    if [[ $cmd == *"kmer"* ]]; then
        echo "  [PID $pid] K-mer Selector (XGBoost/CPU)"
    else
        echo "  [PID $pid] ErrorSmith (CNN/MPS GPU)"
    fi
    echo "    CPU: ${cpu}%  |  Memory: ${mem}%  |  Runtime: ${time}"
done

echo ""
echo "----------------------------------------"
echo "Latest Log Output:"
echo "----------------------------------------"
echo ""

# K-mer Selector log
kmer_log=$(ls -t logs/model_training/kmer_selector_*.log 2>/dev/null | head -1)
if [ -n "$kmer_log" ]; then
    echo "📊 K-mer Selector:"
    tail -3 "$kmer_log" | sed 's/^/    /'
    echo ""
fi

# Base Error Predictor log
base_log=$(ls -t logs/model_training/base_error_predictor_*.log 2>/dev/null | head -1)
if [ -n "$base_log" ]; then
    echo "🧠 Base Error Predictor:"
    tail -3 "$base_log" | sed 's/^/    /'
    echo ""
fi

echo "========================================"
