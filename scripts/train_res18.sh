CUDA_VISIBLE_DEVICES=0 python trainval_net_saba.py \
                   --dataset saba_20171219_train \
                   --net res18 \
                   --epochs 100 \
                   --bs 1 \
                   --nw 1 \
                   --lr 0.0001 \
                   --lr_decay_step 1000000 \
                   --cuda
