#!/bin/bash
#SBATCH --account=def-jte52
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=0-03:00
cd /home/jte52/SENG402
/home/jte52/miniconda3/envs/seal_env/bin/python -m Models.Seals.main --first 2 --input "coco --path /home/jte52/export_coco-instance_segmentsai1_Seal_2022-22_v1.2.json --image_root /home/jte52/2021-22 --split_ratio 70/20/10" --log_dir Models/Seals/log --image_size 512 --eval_split --validation_pause 16 --run_name Seals_2018-19