#!/bin/bash
#SBATCH --account=def-jte52
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=0-03:00
cd /home/jte52/SENG402
/home/jte52/miniconda3/envs/seal_env/bin/python -m scripts.test_model_output