CUDA_VISIBLE_DEVICES=0 python trainval_net_saba.py \
                   --dataset saba_20171219_train \
                   --net alexnet \
                   --epochs 100 \
                   --lr 0.0001 \
                   --lr_decay_step 1000000 \
                   --cuda
