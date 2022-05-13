# LSBATCH: User input
#!/bin/bash
###LSf Syntax
#BSUB -nnodes 1
#BSUB -W 720
#BSUB -G heas
#BSUB -e cars_unroll_lamb_1e-4.txt
#BSUB -o cars_unroll_lamb_1e-4.txt
#BSUB -J cars_unroll_lamb_1e-4
#BSUB -q pbatch
cd /usr/workspace/olivare/
source opence/bin/activate
cd /g/g20/olivare/alpha-beta-sparsity/
 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u car_aircraft_gradient_unroll.py --imagenet_train_data /usr/workspace/RML-data/data/imagenet/imagenet-c.x-full/gaussian_noise/3/ --imagenet_val_data /usr/workspace/RML-data/data/imagenet/val --data /usr/workspace/RML-data/data/car/ --rate 0.2 --save_dir /usr/workspace/olivare/cars_unroll --epoch 95 --worker 4  --lamb 1e-4 --reg-lr 3.5 --dataset cars