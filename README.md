## Installation
Installation on Ubuntu 20.04

### Install NVIDIA GPU Drivers
`sudo apt-get update`

`sudo apt install nvidia-driver-550` (A100)

Check driver installation

`nvidia-smi`

`sudo apt install nvidia-cuda-toolkit`

Install Python 3.11

`sudo add-apt-repository ppa:deadsnakes/ppa`

`sudo apt install python3.11`

`sudo apt install python3.11-dev`

`sudo ln -s /usr/bin/python3.11 /usr/bin/python3`

Install NeMo

`pip install Cython`

`pip install nemo_toolkit[all]` or `python3.11 -m pip install nemo_toolkit[all]`

`pip install --upgrade more-itertools inflect`



