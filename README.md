# Surface match dataset

This dataset contain data which shows value of crossing between two images.

![progress](/docs/surface_match.png)

## How to use

1. Clone this repo.
2. Unzip `/data/data.zip` to get json files with data.
3. Run `/scripts/prepare_data.py` from own folder to get npz file for train your neural network.

## Structure of this repo

### `/data`

This folder contains images and json files which describe data about this images.  
You need unzip `/data/data.zip` to get json files. It require in script `/scripts/prepare_data.py`. 

### `/scripts/prepare_data.py`

This script creates npz file (/train-data/data_224x224.npz) which you can use to train neural network and images (/train-data/images/*.png) if you want read it from disk directly.  
Npz file will contain 2.1 GB and folder with images around 1.5 GB.
