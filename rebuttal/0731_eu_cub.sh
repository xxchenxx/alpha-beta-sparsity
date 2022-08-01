# LSBATCH: User input
#!/bin/bash
###LSf Syntax
#BSUB -nnodes 1
#BSUB -W 720
#BSUB -G heas
#BSUB -e 0731.txt
#BSUB -o 0731.txt
#BSUB -J 0731
#BSUB -q pbatch
cd /usr/workspace/olivare/
source opence/bin/activate
cd /g/g20/olivare/alpha-beta-sparsity/
 
CUDA_VISIBLE_DEVICES=0 nohup python -u cub_finetune_EU.py --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub_eu --epoch 95 --worker 8 --checkpoint /usr/workspace/olivare/imagenetc_imp_debug3/0checkpoint.pth.tar --lr 0.001 > 0731_eu_cub.txt & 

CUDA_VISIBLE_DEVICES=1 nohup python -u cub_finetune_EU.py --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub10_eu --epoch 95 --worker 8 --checkpoint /usr/workspace/olivare/imagenetc_imp_debug3/0checkpoint.pth.tar --lr 0.001 --ten-shot > 0731_eu_cub10.txt & 

CUDA_VISIBLE_DEVICES=2 nohup python -u cub_finetune_mmd.py --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub_mmd --epoch 95 --worker 8 --checkpoint /usr/workspace/olivare/imagenetc_imp_debug3/0checkpoint.pth.tar --lr 0.001 > 0731_mmd_cub.txt & 

CUDA_VISIBLE_DEVICES=3 nohup python -u cub_finetune_mmd.py --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub10_mmd --epoch 95 --worker 8 --checkpoint /usr/workspace/olivare/imagenetc_imp_debug3/0checkpoint.pth.tar --lr 0.001 --ten-shot > 0731_mmd_cub10.txt & 

sleep 12h