# Setup Steps

`./install_conda.sh`

`conda env create -f environment.yaml python=3.8`

`conda activate venv`

`python -m Models.Seals.main --first 2 --input coco\ --path\ /home/fdi19/SENG402/data/annotations/export_coco-instance_segmentsai1_Seal_2022-22_v1.1.json\ --image_root\ /home/fdi19/SENG402/data/images/scott_base/2021-22\ --split_ratio\ 70/0/30 --log_dir ~/new_env/SENG402/Models/Seals/log --train_size 512 --batch_size 8 --validation_pause 16 --run_name Seals_test`
