"""
Test an I2I model.
"""

import argparse

import torch.nn.functional as F
from core.wandb_logger import WandbLogger
from guided_diffusion import dist_util_test, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
from guided_diffusion.testdata import  TestData
import os
import torch.distributed as dist
import clip
from guided_diffusion.test_diff import diffusion_test

os.environ["WANDB_MODE"] = "offline"
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets/test/TH', help='Path of the dataset')
parser.add_argument('--weights', type=str, default='./checkEpoint/model033499_r2v.pt', help='The weight path of pre-trained models')
parser.add_argument('--dire_mode', type=str, default='r2v', help='r2v: Infrared-to-Visible Image Translation; v2r: Visible-to-Infrared Image Translation')
opts = parser.parse_args()

def main(run):
    args = create_argparser().parse_args()
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # set to DETAIL for runtime logging.

    dist_util_test.setup_dist()
    if(dist.get_rank()==0):

        logger.configure(dir='./experiments/log/')
    if(dist.get_rank()==0):
        logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util_test.dev())
    model_weights=args.weights

    model.convert_to_fp16()
    model.load_state_dict(
        dist_util_test.load_state_dict(model_weights, map_location="cpu")
    )
    model.eval()

    val_data = DataLoader(TestData(args.data_dir), batch_size=1, shuffle=False, num_workers=1)  #load_superres_dataval()
    diffusion_test(val_data,model,diffusion, './results/', run , 'test', skip_timesteps=0, iter=0, saveSet=True, mode=args.dire_mode)
    logger.log("All Images Processing Completed!")


def create_argparser():
    defaults = dict(
        data_dir=opts.data_dir,
        weights=opts.weights,
        use_fp16=False,
        dire_mode=opts.dire_mode



    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
if __name__ == "__main__":
    run=WandbLogger()
    main(run)
