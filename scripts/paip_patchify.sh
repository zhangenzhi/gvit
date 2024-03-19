python ./gvit/patchify.py \
        --dataset=paip \
        --datapath=./dataset/paip/output_images_and_masks\
        --resolution=8192\
        --sth=3\
        --split_value=10\
        --target_length=8192 \
        --max_depth=16 \
        --to_size=2