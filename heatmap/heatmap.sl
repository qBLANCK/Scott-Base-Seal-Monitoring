#!/bin/bash
#SBATCH --account=def-fdi19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-01:00
cd /home/fdi19/SENG402 && /home/fdi19/miniconda3/envs/venv/bin/python create_heatmap_vid.py