# #!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo ">>>>>Train Infrared-to-Visible Image Translation<<<<"
CUDA_VISIBLE_DEVICES="0" NCCL_P2P_DISABLE=1  torchrun --nproc_per_node=1 --master_port=4326 ./scripts/train_I2I.py \
                                                                                            --data_dir='./datasets/train' \
                                                                                            --test_dir='./datasets/val' \
                                                                                            --dire_mode='r2v'

echo ">>>>>Train Visible-to-Infrared Image Translation<<<<"
CUDA_VISIBLE_DEVICES="0" NCCL_P2P_DISABLE=1  torchrun --nproc_per_node=1 --master_port=4327 ./scripts/train_I2I.py \
                                                                                            --data_dir='./datasets/train' \
                                                                                            --test_dir='./datasets/val' \
                                                                                            --dire_mode='v2r'


"""

Argument Descriptions:

--data_dir                   # Path to the train datasets.
--test_dir                   # Path to the val datasets.
--dire_mode                  # <r2v> for infrared-to-visible image translation, <v2r> for visible-to-infrared image translation.

"""