#!/bin/bash
#SBATCH --gres=gpu:1                       # Request one gpu
#SBATCH -t 2-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu                            # Partition to run in
#SBATCH --mem=80G                         # Memory total in MiB (for all cores)
#SBATCH -o supcon_all_sweep%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e supcon_all_sweep%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like

# You can change hostname to any command you would like to run
module load miniconda3/23.1.0 gcc/6.2.0
source activate cdt
cd /n/groups/patel/caiwei/2024_MRI/SupContrast

python main_supcon.py --batch_size 32  \
--learning_rate 0.0001  --lr_decay_epochs 200,400,700,800,900 \
--temp 0.5   --cosine --dataset path --mean "(0.09202574,0.08408994,0.07970464)" \
--std "(0.12714559,0.12272861,0.11388139)" \
--data_folder /n/groups/patel/caiwei/Artery/Carotids/CIMT_split/train --size 128

# python -m torch.distributed.launch --nproc_per_node=4 main_supcon.py --batch_size 32  --learning_rate 0.0001  --lr_decay_epochs 200,400,700,800,900 --temp 0.5   --cosine --dataset path --mean "(0.09202574,0.08408994,0.07970464)" --std "(0.12714559,0.12272861,0.11388139)" --data_folder /n/groups/patel/caiwei/Artery/Carotids/CIMT_split/train --size 128

