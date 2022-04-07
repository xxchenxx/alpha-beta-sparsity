cd /usr/workspace/olivare/
source opence/bin/activate
cd /g/g20/olivare/alpha-beta-sparsity


CUDA_VISIBLE_DEVICES=0,1,2,3 python -u cub_finetune_co_tuning.py --imagenet_data /usr/workspace/RML-data/data/imagenet/ --data /usr/workspace/RML-data/data/cub/  --rate 0.2 --pruning_times 10 --save_dir /usr/workspace/olivare/debug --epoch 95 --worker 32 --checkpoint /usr/workspace/olivare/imagenetc_imp_debug3/0checkpoint.pth.tar



NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u cub_finetune_co_tuning.py --imagenet_data /ssd1/xinyu/dataset/imagenet2012/ --data data/  --rate 0.2 --save_dir debug --epoch 95 --worker 4 

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u cub_finetune.py --data data/  --rate 0.2 --save_dir debug --epoch 95 --worker 4 