#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d # Start MPS server

# Define an array to hold the PIDs of the background processes
declare -a pids

mkdir -p logs

# Run multiple processes simultaneously under MPS to load the GPU
for ((i=1; i<=20; i++))
do
    ./a.out > logs/logit_${i}.log 2>&1 &
    pids+=($!)
done

# Initialize a variable to keep track of any failures
any_failures=0

# Wait for all background processes to finish and check their exit statuses
for pid in "${pids[@]}"; do
    wait $pid
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Process with PID $pid exited with a non-zero status: $exit_status"
        any_failures=1
    fi
done

# Check if any process failed
if [ $any_failures -eq 1 ]; then
    echo "At least one process exited with a non-zero status."
else
    echo "All processes exited successfully."
fi

echo quit | nvidia-cuda-mps-control # Shutdown MPS server.
