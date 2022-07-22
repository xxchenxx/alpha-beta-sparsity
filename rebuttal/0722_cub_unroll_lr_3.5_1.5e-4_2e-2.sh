# LSBATCH: User input
#!/bin/bash
###LSf Syntax
#BSUB -nnodes 1
#BSUB -W 720
#BSUB -G heas
#BSUB -e 0722_cub_unroll_lamb_1.5e-4_2e-2.txt
#BSUB -o 0722_cub_unroll_lamb_1.5e-4_2e-2.txt
#BSUB -J 0722_cub_unroll_lamb_1.5e-4_2e-2
#BSUB -q pbatch
cd /usr/workspace/olivare/
source opence/bin/activate
cd /g/g20/olivare/alpha-beta-sparsity/
 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u cub_finetune_gradient_unroll.py --imagenet_train_data /usr/workspace/RML-data/data/imagenet/imagenet-c.x-full/gaussian_noise/3/ --imagenet_val_data /usr/workspace/RML-data/data/imagenet/val --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub10_unroll --epoch 95 --worker 16  --lamb 1.5e-4 --reg-lr 3.5 --lower_steps 1 --checkpoint /usr/workspace/olivare/imagenetc_imp_debug3/0checkpoint.pth.tar --lower-lr 2e-2 --lr 0.001 