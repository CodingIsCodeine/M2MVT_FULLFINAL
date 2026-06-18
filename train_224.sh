#!/bin/bash
#SBATCH --job-name=m2mvt_224
#SBATCH --partition=u22
#SBATCH --nodelist=gnode035
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

python tools/run_net.py \
  --cfg configs/DAAD/M2MVT_DAADX_new.yaml \
  NUM_GPUS 4 \
  DATA.PATH_TO_DATA_DIR /scratch/hriday.samdani/daadx \
  DATA.ANNOTATION_DIR /scratch/hriday.samdani/daadx/annotations \
  OUTPUT_DIR /scratch/hriday.samdani/m2mvt_run_224 \
  TRAIN.CHECKPOINT_FILE_PATH /scratch/hriday.samdani/m2mvt_run/checkpoints/checkpoint_epoch_00008.pyth \
  TRAIN.AUTO_RESUME False \
  DATA.NUM_FRAMES 16 \
  DATA.TRAIN_JITTER_SCALES [256,320] \
  DATA.TRAIN_CROP_SIZE 224 \
  DATA.TEST_CROP_SIZE 224 \
  TRAIN.BATCH_SIZE 4 \
  DATA_LOADER.NUM_WORKERS 4 \
  DATA_LOADER.PERSISTENT_WORKERS True
