# SENG402 Capstone Project
## Scott Base Seal Monitoring
> **Supervisors:** [Oliver Batchelor](https://ucvision.org.nz/oliver-batchelor/) & [Richard Green](https://www.canterbury.ac.nz/engineering/contact-us/people/richard-green.html)
> **Partner:** [Antarctica NZ](https://www.google.com/search?client=safari&rls=en&q=antarctica%20nz&ie=UTF-8&oe=UTF-8)
> 
> The goal of this project was to monitor Weddel seal activity during the Scott Base rebuild at Antartica (starting 2022/2023 summer) to ensure that they are minimally disrupted by building activity. Over the 2021/2022 summer, two 180Mp cameras were successfully installed at Pram Point and Turtle Rock, capturing continuous time-lapse image data of seals from November to February. With support from our deep learning team, this data was analysed for seal count/movemen to ensure this tech is fit-for-purpose for monitoring impact on seals of the Scott Base rebuild that starts late 2022.

<p align="center"><img width="50%" src="data/figures/weddell%20seal.jpeg"></p>

|  |  |
|--|--|
| **Seal Counts** | ![Seal Counts](/data/figures/Seal_Counts.png) |
| **Scott Base** | ![Scott Base](/data/figures/scott_base.jpg) |
| **Heatmap Example Image** | ![Heatmap](/data/figures/heatmap.png) |

|**Scott Base Original**|**Turtle Rock Original**|
|--|--|
|![Scott Base Original](/data/figures/og_scott_base.png)|![Turtle Rock Original](/data/figures/og_turtle_rock.png)|


## Project Objectives
 - [x] Recieve and store large RAW images from two time lapse cameras from Nov 2021 — Feb 2022
 - [x] Design & Train an accurate CNN model to detect Weddel Seals from afar
 - [x] Use the trained model to generate Seal counts for each image
	 - [x] Plot Seal counts over the season (including the timestamp)
	 - [ ] Extract key metrics (movement pattens over season)
 - [x] Condense images into a short (~5 Minutes) Timelapse
 - [x] Combine timelapse and Seal detections to create a Heatmap Timelapse showing Movement
 - [ ] Statistical analysis on the raw data, including an identification and analysis of possible disturbance events.
 - [ ] Provide a report including a description of the seal population, and summary assessment based on the statistics obtained through analysis. This report is to include tables, maps, and images as appropriate to support the Scott Base Redevelopment environmental monitoring.
	 - [ ] Provide an advice note containing recommendations for the next summer season based on the previous summer season results (include cost, power, logistics/support) – with the additional goals of:
		 - Real-time alert for seal disturbance by construction noise
		 - Accurate 3D seal location
		 - Identification of individual seals using available biometric data


### Setup Steps

`./install_conda.sh`

`conda env create -f environment.yaml python=3.8`

`conda activate venv`

`python -m Models.Seals.main --first 2 --input coco\ --path\ /home/fdi19/SENG402/data/annotations/export_coco-instance_segmentsai1_Seal_2022-22_v1.1.json\ --image_root\ /home/fdi19/SENG402/data/images/scott_base/2021-22\ --split_ratio\ 70/0/30 --log_dir ~/new_env/SENG402/Models/Seals/log --image_size 512 --batch_size 8 --validation_pause 16 --run_name Seals_test`

write vids, model and images to /csse/research/antarctica_seals
add insructions to access the data (using lab machine) and the sstucture of the dir