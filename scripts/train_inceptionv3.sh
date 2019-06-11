CUDA_VISIBLE_DEVICES=0 python trainval_net_inceptionv3.py \
                   --dataset saba_20171219_train \
                   --net inceptionv3 \
                   --epochs 50 \
                   --bs 1 \
                   --nw 1 \
                   --lr 0.00001 \
                   --lr_decay_step 1000000 \
                   --cuda
