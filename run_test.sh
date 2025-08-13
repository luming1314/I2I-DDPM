# #!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo ">>>>>Test Infrared-to-Visible Image Translation<<<<"
CUDA_VISIBLE_DEVICES="0" python ./scripts/test_I2I.py \
                                --data_dir='./datasets/test/TH' \
                                --weights='./checkpoint/model033499_r2v.pt' \
                                --dire_mode='r2v'

echo ">>>>>Test Visible-to-Infrared Image Translation<<<<"
CUDA_VISIBLE_DEVICES="0" python ./scripts/test_I2I.py \
                                --data_dir='./datasets/test/VIS' \
                                --weights='./checkpoint/model033499_v2r.pt' \
                                --dire_mode='v2r'


"""

Argument Descriptions:

--data_dir                   # Path to the input images.
--weights                    # Path to the pre-trained model file.
--dire_mode                  # <r2v> for infrared-to-visible image translation, <v2r> for visible-to-infrared image translation.

"""
