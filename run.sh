#!/bin/sh
echo 'This training process takes around 24 hours, please be patient and do not interrupt it...'

#python train.py --train dense161
CUDA_VISIBLE_DEVICES=1 python train.py --train dense121
CUDA_VISIBLE_DEVICES=1 python train.py --train dense201
CUDA_VISIBLE_DEVICES=1 python train.py --train dense169
CUDA_VISIBLE_DEVICES=1 python train.py --train res50
CUDA_VISIBLE_DEVICES=1 python train.py --train res101
CUDA_VISIBLE_DEVICES=1 python train.py --train res152
CUDA_VISIBLE_DEVICES=1 python train.py --train inceptionv3
CUDA_VISIBLE_DEVICES=1 python train.py --train vgg19bn
CUDA_VISIBLE_DEVICES=1 python train.py --train vgg16bn

echo 'Training finished.'
