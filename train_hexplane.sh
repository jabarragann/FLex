#!/bin/bash
# This script installs environment
#SBATCH --job-name=train_hexplane
#SBATCH --nodes=1
#SBATCH --partition=gpu                # This is currently the only partition
#SBATCH --gres=shard:35                # Change this to the desired number of GPUs or SHARDs
#SBATCH --time=UNLIMITED                # Change this to the desired max runtime (hh:mm:ss)
 
# optionally define placeholders that can be overwritten from the command line
config_file="config/stereomis/Hexplane_p2_8_1_example.yaml" # change config here as otherwise not working atm.
 
# software setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flex
 
# do stuff
python main.py config="$config_file"