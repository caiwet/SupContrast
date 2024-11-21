#!/bin/bash
#SBATCH --gres=gpu:1,vram:48G                       # Request one gpu
#SBATCH -t 2-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                            # Partition to run in
#SBATCH --mem=40G                         # Memory total in MiB (for all cores)
#SBATCH -o ce_sex_train%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ce_sex_train%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like

# You can change hostname to any command you would like to run
module load miniconda3/23.1.0 gcc/9.2.0 cuda/12.1
source activate cdt
cd /n/groups/patel/caiwei/2024_MRI/SupContrast


python eval.py --batch_size 64 --dataset 'path' --size 128 \
 --mean "(0.456, 0.456, 0.456)"   --std "(0.224, 0.224, 0.224)"  \
 --data_folder /n/groups/patel/caiwei/Artery/Carotids/CIMT_split/val \
 --n_cls 3 --ckpt /n/groups/patel/caiwei/2024_MRI/SupContrast/save/linear_best_model.pth \
 --ckpt_cls /n/groups/patel/caiwei/2024_MRI/SupContrast/save/linear_best_model_classifier.pth

