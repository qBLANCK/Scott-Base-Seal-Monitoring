# Setup Steps

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

`conda env create -f environment.yaml python=3.8` might not need the python=3.8 as in yaml

`conda activate venv`

`python -m Models.Seals.main --first 2 --input coco\ --path\ /home/fdi19/SENG402/data/annotations/export_coco-instance_segmentsai1_Seal_2022-22_v1.1.json\ --image_root\ /home/fdi19/SENG402/data/images/scott_base/2021-22\ --split_ratio\ 70/0/30 --log_dir ~/new_env/SENG402/Models/Seals/log --train_size 512 --batch_size 8 --validation_pause 16 --run_name Seals_test`

write vids, model and images to /csse/research/antarctica_seals
add insructions to access the data (using lab machine) and the sstucture of the dir