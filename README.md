# Scott Base Seal Monitoring

<p  align="center"><img  width="50%"  src="data/figures/weddell%20seal.jpeg"></p>

> **Supervisors:** [Oliver Batchelor](https://ucvision.org.nz/oliver-batchelor/) & [Richard Green](https://www.canterbury.ac.nz/engineering/contact-us/people/richard-green.html)

> **Partner:** [Antarctica NZ](https://www.google.com/search?client=safari&rls=en&q=antarctica%20nz&ie=UTF-8&oe=UTF-8)

>

> The goal of this project was to monitor Weddel seal activity during the Scott Base rebuild at Antartica (starting 2022/2023 summer) to ensure that they are minimally disrupted by building activity. Over the 2021/2022 summer, two 180Mp cameras were successfully installed at Pram Point and Turtle Rock, capturing continuous time-lapse image data of seals from November to February. With support from our deep learning team, this data was analysed for seal count/positioning to ensure this tech is fit-for-purpose for monitoring impact on seals of the Scott Base rebuild that starts late 2022.

## Method

This solution applies a RetinaNet CNN to detect the small and dense seals. The object detector can detect individual seals and a pairing of seals. Additionally, a second CNN was developed to detect low-visability inducing Antractic Snowstorms. These snowstorms greatly affect the validity of the seal detection CNN as they objstuct the view of the seals. Detection of Snowstorms allows for detection of false low-counts.

## Artifacts & Examples

|                     |                                               |
| ------------------- | --------------------------------------------- |
| **Seal Counts**     | ![Seal Counts](/data/figures/Seal_Counts.png) |
| **Scott Base**      | ![Scott Base](/data/figures/scott_base.jpg)   |
| **Heatmap Example** | ![Heatmap](/data/figures/heatmap.gif)         |

| **Scott Base Original**                                 | **Turtle Rock Original**                                  |
| ------------------------------------------------------- | --------------------------------------------------------- |
| ![Scott Base Original](/data/figures/og_scott_base.png) | ![Turtle Rock Original](/data/figures/og_turtle_rock.png) |

## Project Objectives

- [x] Recieve and store large RAW images from two time lapse cameras from Nov 2021 — Feb 2022

- [x] Design & Train an accurate CNN model to detect Weddel Seals from afar

- [x] Use the trained model to generate Seal counts for each image

- [x] Plot Seal counts over the season (including the timestamp)

- [x] Condense images into a short (~5 Minutes) Timelapse

- [x] Combine timelapse and Seal detections to create a Heatmap Timelapse showing Movement

- [x] Provide a report including a description of the seal population, and summary assessment based on the statistics obtained through analysis. This report is to include tables, maps, and images as appropriate to support the Scott Base Redevelopment environmental monitoring.

- [x] Provide an advice note containing recommendations for the next summer season based on the previous summer season results (include cost, power, logistics/support) – with the additional goals of:

  - Real-time alert for seal disturbance by construction noise
  - Accurate 3D seal location
  - Identification of individual seals using available biometric data

## Future Scope

- Mathematically model the seal counts as a function of time. (Usefull for detecting disturbance of colony)

- Statistical analysis on the raw data, including an identification and analysis of possible disturbance events.

## Codebase Map

```
├── data
│   ├── counts                    Seal counts for seasons ('a','b','c' are camera
│   │                               viewpoints and 's' includes snowstorm detection data).
│   └── figures                   Illustrative graphics on project and Seal data.
├── libs
│   ├── convert                   Inc. script for converting RAW images to jpg & requried
│   │                               dcraw library.
│   ├── heatmappy                 Lib for generating Heatmap videos.
│   └── tools                     Useful tools for CNN Training.
└── Models
    ├── Seals                     Weddel Seal Object Detection training (RetinaNet).
    │   ├── dataset               Helper functions splitting images into train, test &
    │   │   │                       validation datasets.
    │   │   └── imports           Logic for importing images and their annotations (coco
    │   │                            & json format).
    │   └── detection             Helper functions for model evaluation, visualisation and
    │   │   │                       model implementation.
    │   │   └── retina            RetinaNet CNN implementation.
    │   └── models                Abstract models for Implementation & Evaluation
    └── Snowstorm                 Antarctic Snowstorm Classification (ResNet-18).
```

## Setup Steps

This project has been deleveloped with a Conda environment for ease of reproduction. If you do not have conda installed, run `./install_conda.sh`.

To install the conda environment with all required dependencies run `conda env create -f environment.yaml python=3.8`.

To activate the environment run `conda activate seal_env`.

## Training Seal model

The main script (used for training) can be run with `python -m Models.Seals.main`. Input parameters, their descriptions and defaults can be viewed by adding the `-h` or `--help`.

Example below:

```
python -m Models.Seals.main --first 2 --input coco\ --path annotations/annotation.json --image_root /csse/reserach/.../2021-22 --split_ratio 70/15/15 --log_dir Models/Seals/log --image_size 512 --batch_size 8 --validation_pause 16 --run_name Seals_test
```

## Large Project Artifacts

Other large artifacts such as models, training logs, annotations, and images that are too large for GitHub are stored on the University of Canterbury's csse drive (`/csse/research/antarctica_seals`).

Below is the tree map for the artifacts folder:

```
├── annotations
├── images
│   ├── scott_base
│   │   ├── 2018-19               Including the three viewpoints (A, B & C).
│   │   │   ├── CameraA
│   │   │   ├── CameraB
│   │   │   └── CameraC
│   │   ├── 2019-20               Including the three viewpoints (A, B & C).
│   │   │   ├── CameraA
│   │   │   ├── CameraB
│   │   │   └── CameraC
│   │   └── 2021-22
│   ├── training_sets             Training images for 2018-19 and 2019-20 seasons. Used
│   │   │                           with Oliver's annnnotation format.
│   │   ├── scott_base
│   │   └── turtle_rock
│   └── turtle_rock
│       └── 2021-22
├── models
│   ├── seal_od                   Dir for Seal object detection models.
│   │   ├── olivers
│   │   │   ├── 2018-19
│   │   │   └── 2019-20
│   │   ├── Seals_2018-19
│   │   └── Seals_2021-22
│   └── snowstorm_cl              Dir for Snowstorm classification models.
└── videos                        Inc. Timelapse and Heatmap of 2021-22 Scott Base.
```
