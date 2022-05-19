# LSBATCH: User input
#!/bin/bash
###LSf Syntax
#BSUB -nnodes 1
#BSUB -W 720
#BSUB -G heas
#BSUB -e cub_mean_teacher_noise10.txt
#BSUB -o cub_mean_teacher_noise10.txt
#BSUB -J cub_mean_teacher_noise10
#BSUB -q pbatch
cd /usr/workspace/olivare/
source opence/bin/activate
cd /g/g20/olivare/alpha-beta-sparsity/
 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u cub_finetune_consistent.py --imagenet_train_data /usr/workspace/RML-data/data/imagenet/imagenet-c.x-full/gaussian_noise/3/ --imagenet_val_data /usr/workspace/RML-data/data/imagenet/val --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub_unroll_consistency --epoch 95 --worker 4 --lr 0.1 --ten-shot