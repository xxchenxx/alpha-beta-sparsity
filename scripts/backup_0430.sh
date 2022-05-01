CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 nohup python -u cub_finetune_gdp.py --imagenet_train_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --imagenet_val_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ --data data/  --rate 0.2 --save_dir cub_2 --epoch 95 --worker 16  --dist-url tcp://127.0.0.1:36606 > with_img.out &


CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE=1 python -u cub_finetune_gradient_unroll.py --imagenet_train_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --imagenet_val_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ --data data/  --rate 0.2 --save_dir cub_2 --epoch 95 --worker 16  --dist-url tcp://127.0.0.1:36606


CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE=1 nohup python -u cub_finetune_gdp.py --imagenet_train_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --imagenet_val_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ --data data/  --rate 0.2 --save_dir cub_lr_5 --epoch 95 --worker 16  --dist-url tcp://127.0.0.1:36606 --lamb 1e-4 --reg-lr 5 > lr5.out &

CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 nohup python -u cub_finetune_gdp.py --imagenet_train_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --imagenet_val_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ --data data/  --rate 0.2 --save_dir cub_lr_2 --epoch 95 --worker 16  --dist-url tcp://127.0.0.1:36607 --lamb 1e-4 --reg-lr 2 > lr2.out &

CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 nohup python -u cub_finetune_gdp.py --imagenet_train_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --imagenet_val_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ --data data/  --rate 0.2 --save_dir cub_lr_3 --epoch 95 --worker 16  --dist-url tcp://127.0.0.1:36607 --lamb 1e-4 --reg-lr 3 > lr3.out &

CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 nohup python -u cub_finetune_gdp.py --imagenet_train_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --imagenet_val_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ --data data/  --rate 0.2 --save_dir cub_lr_5 --epoch 95 --worker 16  --dist-url tcp://127.0.0.1:36607 --lamb 1e-4 --reg-lr 5 > lr5.out &

CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 nohup python -u cub_finetune_gdp.py --imagenet_train_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --imagenet_val_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ --data data/  --rate 0.2 --save_dir cub_lr_4 --epoch 95 --worker 16  --dist-url tcp://127.0.0.1:36607 --lamb 1e-4 --reg-lr 4 > lr4.out &



CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE=1 nohup python -u cub_finetune_gradient_unroll.py --imagenet_train_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --imagenet_val_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ --data data/  --rate 0.2 --save_dir cub_unroll_lr_5 --epoch 95 --worker 16  --dist-url tcp://127.0.0.1:37706 --lamb 1e-4 --reg-lr 5 &


CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 python -u test_gdp.py --imagenet_train_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --imagenet_val_data /ssd1/xinyu/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ --data data/  --rate 0.2 --save_dir cub_unroll_lr_5 --epoch 95 --worker 16  --dist-url tcp://127.0.0.1:37606 --lamb 1e-4 --reg-lr 5