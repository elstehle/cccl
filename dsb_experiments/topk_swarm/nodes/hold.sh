#!/usr/bin/env bash
# topk_swarm hold job. Logs node facts to ~/b200_alloc/job_<jobid>.log, then sleeps so the
# allocation stays until scancel or the time limit. Submitted by tools/broker.py request as
# ~/b200_alloc/topk_hold.sh. At job end (TERM from slurm) it removes the swarm docker container
# on computelab so no process keeps a reference to a GPU that slurm hands to the next job.
# Attach:  ssh <node>   or   srun --jobid=<id> --overlap --pty bash -l
# Release: scancel <id>
L="${HOME}/b200_alloc/job_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$L")"
{
  echo "job $SLURM_JOB_ID started $(date) on $(hostname) partition=$SLURM_JOB_PARTITION"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  cpus=$(nproc)  arch=$(uname -m)"
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,power.limit --format=csv 2>&1
} > "$L" 2>&1
cleanup() {
  trap - TERM INT EXIT
  if command -v docker >/dev/null 2>&1 && docker inspect topk_swarm >/dev/null 2>&1; then
    docker rm -f topk_swarm >> "$L" 2>&1
  fi
  echo "job $SLURM_JOB_ID ending $(date)" >> "$L"
  exit 0
}
trap cleanup TERM INT EXIT
sleep infinity &
wait $!
cleanup
