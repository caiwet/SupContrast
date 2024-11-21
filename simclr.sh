#!/bin/bash
#SBATCH --gres=gpu:1, vram:48G                       # Request one gpu
#SBATCH -t 0-12:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                           # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)
#SBATCH -o simclr_all_train%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e simclr_all_train%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like

# You can change hostname to any command you would like to run
module load miniconda3/23.1.0 gcc/6.2.0
source activate cdt
cd /n/groups/patel/caiwei/2024_MRI/SupContrast

python main_supcon.py --batch_size 32  \
--learning_rate 3e-4  --lr_decay_epochs 200,400,700,800,900 \
--temp 0.5   --cosine --dataset path --mean "(0.485, 0.456, 0.406)" \
--std "(0.229, 0.224, 0.225)" \
--data_folder /n/groups/patel/caiwei/Artery/Carotids/CIMT_split/train --size 128 \
--method SimCLR

# python -m torch.distributed.launch --nproc_per_node=4 main_supcon.py --batch_size 32  --learning_rate 0.0001  --lr_decay_epochs 200,400,700,800,900 --temp 0.5   --cosine --dataset path --mean "(0.09202574,0.08408994,0.07970464)" --std "(0.12714559,0.12272861,0.11388139)" --data_folder /n/groups/patel/caiwei/Artery/Carotids/CIMT_split/train --size 128

