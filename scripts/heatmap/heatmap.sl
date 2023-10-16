#!/bin/bash
#SBATCH --account=def-jte52
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=0-03:00
cd /home/jte52/SENG402/scripts/heatmap
/home/jte52/miniconda3/envs/seal_env/bin/python  -m create_heatmap --chunks 4 --timelapse "2022-23_timelapse.mp4"  --seals "Locations_2022-23_gen-Oct15_filtered.csv" --frames 4034