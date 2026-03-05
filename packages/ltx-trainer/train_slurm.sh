#!/bin/bash
#SBATCH --job-name=ltx2-sdr-to-hdr
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/ltx2_sdr_to_hdr/slurm_%j.log
#SBATCH --error=outputs/ltx2_sdr_to_hdr/slurm_%j.log

# ── Environment ────────────────────────────────────────────────────────────────
export PATH="/home/eli/.local/bin:$PATH"   # uv
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prevent tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAINER_DIR="/scratch/eli/disney/LTX-2/packages/ltx-trainer"
CONFIG="configs/ltx2_v2v_ic_lora_s2r_hdr.yaml"
ACCELERATE_CONFIG="configs/accelerate/fsdp.yaml"

# ── Run ────────────────────────────────────────────────────────────────────────
cd "$TRAINER_DIR"

# Create output dir so SLURM can write the log before the trainer does
mkdir -p outputs/ltx2_sdr_to_hdr

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo "Started:  $(date)"
echo "──────────────────────────────────────────"

uv run accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    scripts/train.py "$CONFIG"

echo "──────────────────────────────────────────"
echo "Finished: $(date)"
