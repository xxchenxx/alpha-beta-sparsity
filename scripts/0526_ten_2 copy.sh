# LSBATCH: User input
#!/bin/bash
###LSf Syntax
#BSUB -nnodes 1
#BSUB -W 720
#BSUB -G heas
#BSUB -e 0526_cub10_unroll_lr10_lamb_1e-4_2.txt
#BSUB -o 0526_cub10_unroll_lr10_lamb_1e-4_2.txt
#BSUB -J 0526_cub10_unroll_lr10_lamb_1e-4_2
#BSUB -q pbatch
cd /usr/workspace/olivare/
source opence/bin/activate
cd /g/g20/olivare/alpha-beta-sparsity/
 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u cub_finetune_gradient_unroll.py --imagenet_train_data /usr/workspace/RML-data/data/imagenet/train --imagenet_val_data /usr/workspace/RML-data/data/imagenet/val --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub_unroll --epoch 95 --worker 4  --lamb 1e-4 --reg-lr 10 --imagenet-pretrained --lower_steps 5 --ten-shot --seed 2