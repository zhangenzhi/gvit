python ./gvit/patchify.py \
        --dataset=paip \
        --datapath=./dataset/paip/output_images_and_masks\
        --resolution=4096\
        --sth=7\
        --split_value=100\
        --target_length=4096 \
        --max_depth=16 \
        --to_size=8