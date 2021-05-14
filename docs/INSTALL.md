## Installation
This installation guide shows you how to set up the environment for running our code using conda or Singularity container.

First clone the ActiveMRI repository
```
git clone https://github.com/tianweiy/SeqMRI.git
cd SeqMRI
```
Then start a virtual environment with new environment variables nad
```
conda create --name activemri python=3.7
conda activate activemri 
```
Install PyTorch 
```
conda install pytorch=1.7.1 torchvision cudatoolkit=10.2 -c pytorch
```
Install all requirements
```
pip install -r requirements.txt
```
## Singularity Container
To use the Singularity Container, run the following piece of code before creating the virtual environment.

Login and enter API access token
```
singularity remote login
```
Build the image to a .sif file
```
singularity build --remote active-mri.sif active-mri.def
```
Run a singularity shell 
```
singularity shell --nv active-mri.sif
```
Proceed with the installation instruction above.