#!/bin/bash
#SBATCH --gres=gpu:1,vram:48G                       # Request one gpu
#SBATCH -t 2-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                           # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)
#SBATCH -o linear_train%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e linear_train%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like

# You can change hostname to any command you would like to run
module load miniconda3/23.1.0 gcc/9.2.0 cuda/12.1
source activate cdt
cd /n/groups/patel/caiwei/2024_MRI/SupContrast

python main_linear.py --batch_size 64 \
  --learning_rate 0.0001 --lr_decay_epochs 70,80,90,150,200,250,300,350,400,450,500 \
  --lr_decay_rate 0.4533962493340439 --momentum 0.8774814333092061 \
  --weight_decay 0.2728030166414763 --epochs 1000 \
  --dataset 'path' --size 224 \
  --mean "(0.456, 0.456, 0.456)" \
  --std "(0.224, 0.224, 0.224)" \
  --data_folder /n/groups/patel/caiwei/Artery/Carotids/CIMT_split --n_cls 3 \
  --ckpt /n/groups/patel/caiwei/2024_MRI/model/save/simclr_best_model.pth