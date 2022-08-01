# LSBATCH: User input
#!/bin/bash
###LSf Syntax
#BSUB -nnodes 1
#BSUB -W 720
#BSUB -G heas
#BSUB -e 0727_eu_cub.txt
#BSUB -o 0727_eu_cub.txt
#BSUB -J 0727_eu_cub
#BSUB -q pbatch
cd /usr/workspace/olivare/
source opence/bin/activate
cd /g/g20/olivare/alpha-beta-sparsity/
 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u cub_finetune_EU.py --data /usr/workspace/RML-data/data/cub/ --rate 0.2 --save_dir /usr/workspace/olivare/cub_eu --epoch 95 --worker 16 --checkpoint /usr/workspace/olivare/imagenetc_imp_debug3/0checkpoint.pth.tar --lr 0.001 