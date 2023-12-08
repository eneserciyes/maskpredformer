#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=train_maskpredformer
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me2646@nyu.edu

module purge

singularity exec --nv \
            --bind /scratch \
	    --overlay /scratch/me2646/maskpredformer.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python train_predictor.py"
