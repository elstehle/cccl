#!/usr/bin/env bash
# topk_swarm hold job. Logs node facts to ~/b200_alloc/job_<jobid>.log, then sleeps so the
# allocation stays until scancel or the time limit. Submitted by tools/broker.py request.
# Attach:  ssh <node>   or   srun --jobid=<id> --overlap --pty bash -l
# Release: scancel <id>
L="${HOME}/b200_alloc/job_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$L")"
{
  echo "job $SLURM_JOB_ID started $(date) on $(hostname) partition=$SLURM_JOB_PARTITION"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  cpus=$(nproc)  arch=$(uname -m)"
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,power.limit --format=csv 2>&1
} > "$L" 2>&1
sleep infinity
