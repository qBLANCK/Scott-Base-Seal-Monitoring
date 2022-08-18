# Setup Steps

`conda env create -f environment.yaml` might not need the python=3.8 as in yaml

`conda activate venv`
`conda install -n venv ipykernel --update-deps --force-reinstall'

<!-- Do we need this? -->

cd into Seals
`conda install --file requirements.txt`
`pip install -r requirements.txt` for the pytorch tools (need to move into this repo)

run with `python -m Models.Seals.main.py`
