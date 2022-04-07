# LSBATCH: User input
#!/bin/bash
###LSf Syntax
#BSUB -nnodes 1
#BSUB -W 720
#BSUB -G heas
#BSUB -e cub_co_tuning_4e-5_beta_only.txt
#BSUB -o cub_co_tuning_4e-5_beta_only.txt
#BSUB -J cub_co_tuning_4e-5_beta_only
#BSUB -q pbatch
cd /usr/workspace/olivare/
source opence/bin/activate
cd /g/g20/olivare/alpha-beta-sparsity/
 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u cub_finetune_co_tuning.py --imagenet_train_data /usr/workspace/RML-data/data/imagenet/train --imagenet_val_data /usr/workspace/RML-data/data/imagenet/val --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub_co_tuning_4e-5_beta_only --epoch 95 --worker 4 --checkpoint /usr/workspace/olivare/imagenetc_imp_debug3/0checkpoint.pth.tar --l1-reg-beta 4e-5