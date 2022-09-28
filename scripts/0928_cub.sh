# LSBATCH: User input
#!/bin/bash
###LSf Syntax
#BSUB -nnodes 1
#BSUB -W 720
#BSUB -G heas
#BSUB -e cub_mix.txt
#BSUB -o cub_mix.txt
#BSUB -J cub_mix
#BSUB -q pbatch
cd /usr/workspace/olivare/
source opence/bin/activate
cd /g/g20/olivare/alpha-beta-sparsity/
 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u cub_finetune_mix.py --imagenet_train_data /usr/workspace/RML-data/data/imagenet/imagenet-c.x-full/gaussian_noise/3/ --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub_mix --epoch 95 --worker 4 --rate 0.2 --lr 0.01 --pruning_times 1
