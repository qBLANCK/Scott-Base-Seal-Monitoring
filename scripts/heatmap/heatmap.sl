#!/bin/bash
#SBATCH --account=def-jte52
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-01:00
cd /home/jte52/SENG402 && /home/jte52/miniconda3/envs/venv/bin/python create_heatmap_vid.py