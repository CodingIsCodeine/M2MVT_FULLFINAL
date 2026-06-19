#!/bin/bash
#SBATCH --job-name=m2mvt_224
#SBATCH --partition=u22
#SBATCH --nodelist=gnode044
#SBATCH --time=24:00:00
#SBATCH -A hriday.samdani
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --mem-per-cpu=2G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd ~/M2MVT_FULLFINAL
source ~/venvs/m2mvt/bin/activate

mkdir -p /scratch/hriday.samdani/m2mvt_run_224

echo "===== JOB INFO ====="
hostname
date
nvidia-smi
echo "===================="

NCCL_P2P_DISABLE=1 python tools/run_net.py \
  --cfg configs/DAAD/M2MVT_DAADX_new.yaml \
  --opts \
  NUM_GPUS 4 \
  DATA.PATH_TO_DATA_DIR /scratch/hriday.samdani/daadx \
  DATA.ANNOTATION_DIR /scratch/hriday.samdani/daadx/annotations \
  OUTPUT_DIR /scratch/hriday.samdani/m2mvt_run_224 \
  DATA.NUM_FRAMES 16 \
  DATA.TRAIN_JITTER_SCALES [256,320] \
  DATA.TRAIN_CROP_SIZE 224 \
  DATA.TEST_CROP_SIZE 224 \
  TRAIN.BATCH_SIZE 4 \
  TRAIN.MIXED_PRECISION True \
  TRAIN.CHECKPOINT_FILE_PATH /scratch/hriday.samdani/m2mvt_run/checkpoints/checkpoint_epoch_00020.pyth \
  TRAIN.CHECKPOINT_EPOCH_RESET True \
  DATA_LOADER.NUM_WORKERS 4 \
  DATA_LOADER.PERSISTENT_WORKERS True


# NCCL_P2P_DISABLE=1 python tools/run_net.py \
#   --cfg configs/DAAD/M2MVT_DAADX_new.yaml \
#   --opts \
#   NUM_GPUS 4 \
#   DATA.PATH_TO_DATA_DIR /scratch/hriday.samdani/daadx \
#   DATA.ANNOTATION_DIR /scratch/hriday.samdani/daadx/annotations \
#   OUTPUT_DIR /scratch/hriday.samdani/daadx_runs/k400_pretrain \
#   DATA.NUM_FRAMES 16 \
#   DATA.TRAIN_JITTER_SCALES [256,320] \
#   DATA.TRAIN_CROP_SIZE 224 \
#   DATA.TEST_CROP_SIZE 224 \
#   TRAIN.BATCH_SIZE 4 \
#   TRAIN.MIXED_PRECISION True \
#   TRAIN.CHECKPOINT_FILE_PATH /scratch/hriday.samdani/MViTv2_S_16x4_k400_f302660347.pyth \
#   TRAIN.CHECKPOINT_EPOCH_RESET True \
#   TRAIN.AUTO_RESUME False \
#   DATA_LOADER.NUM_WORKERS 4 \
#   DATA_LOADER.PERSISTENT_WORKERS True