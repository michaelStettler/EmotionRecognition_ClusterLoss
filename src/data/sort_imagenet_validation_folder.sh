#!/usr/bin/env bash

folder_path='../../../../Downloads/ImageNet/'

index=1

while read -r p; do
    # -p allows to create the folder only if needed
    mkdir -p $folder_path'validation/'$p
    # pad number with 0
    printf -v tmp_idx '%08d' $index
    mv $folder_path'validation/ILSVRC2012_val_'$tmp_idx'.JPEG' $folder_path'validation/'$p
    echo $p $tmp_idx
    ((index++))
done < $folder_path'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
