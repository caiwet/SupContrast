#!/bin/bash
#SBATCH --gres=gpu:1,vram:48G                       # Request one gpu
#SBATCH -t 2-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                            # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)
#SBATCH -o ce_sweep_pretrained%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ce_sweep_pretrained%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like

# You can change hostname to any command you would like to run
module load miniconda3/23.1.0 gcc/6.2.0
source activate cdt
cd /n/groups/patel/caiwei/2024_MRI/SupContrast
python main_ce_sweep.py --batch_size 32   \
  --cosine --dataset path --mean "(0.485, 0.456, 0.406)" \
  --std "(0.229, 0.224, 0.225)" \
  --data_folder /n/groups/patel/caiwei/Artery/Carotids/CIMT_split --n_cls 3 --size 128

