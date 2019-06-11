CUDA_VISIBLE_DEVICES=0 python trainval_net_saba.py \
                   --dataset saba_20171219_local_train \
                   --net vgg16 \
                   --epochs 100 \
                   --bs 1 \
                   --nw 1 \
                   --lr 0.001 \
                   --lr_decay_step 1000000 \
                   --cuda \
                   --save_dir models_local
