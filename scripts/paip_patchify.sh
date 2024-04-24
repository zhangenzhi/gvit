python ./gvit/patchify.py \
        --dataset=paip \
        --datapath=./dataset/paip/output_images_and_masks\
        --resolution=8192\
        --sth=11\
        --split_value=20\
        --target_length=4096 \
        --max_depth=16 \
        --to_size=2