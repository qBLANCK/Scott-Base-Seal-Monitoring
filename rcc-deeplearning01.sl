#!/bin/bash
#SBATCH --account=def-jte52
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=0-03:00
cd /home/jte52/SENG402
/home/jte52/miniconda3/envs/seal_env/bin/python -m Models.Seals.main --first 2 --input "json --path /home/jte52/scott_base_no_pair.json" --second_input "coco --path /home/jte52/annotations_new.json --image_root /home/jte52/images/2021-22" --log_dir Models/Seals/log --validation_pause 16 --run_name Seals02
