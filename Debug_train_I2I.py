"""
Train an I2I model.
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
from guided_diffusion.valdata import  ValData
import os
import torch.distributed as dist

os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets/train', help='Path of the train dataset')
parser.add_argument('--test_dir', type=str, default='./datasets/val', help='Path of the val dataset')
parser.add_argument('--dire_mode', type=str, default='r2v', help='r2v: Infrared-to-Visible Image Translation; v2r: Visible-to-Infrared Image Translation')

opts = parser.parse_args()

def main(run):
    args = create_argparser().parse_args()
    args.test_dir = args.test_dir + '/' + 'TH/' if args.dire_mode == 'r2v' else args.test_dir + '/' + 'VIS/'
    print(f'tets_dir == {args.test_dir}')

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
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    if(dist.get_rank()==0):
        logger.log("creating data loader...")

    data_dir1=os.path.join(args.data_dir, 'TH/' if args.dire_mode == 'r2v' else 'VIS/')
    gt_dir=os.path.join(args.data_dir, 'VIS/' if args.dire_mode == 'r2v' else 'TH/')

    val_data = DataLoader(ValData(args.test_dir), batch_size=1, shuffle=False, num_workers=1)  #load_superres_dataval()
    data = load_superres_data(
        data_dir1,
        gt_dir,
        args.batch_size,
        image_size=128,
    )
    if(dist.get_rank()==0):
        logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        val_dat=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        test_interval=args.test_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop(run)


def load_superres_data(data_dir,gt_dirs, batch_size, image_size):
    data = load_data(
        data_dir=data_dir,
        gt_dir=gt_dirs,
        batch_size=batch_size,
        image_size=image_size,

    )
    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs

def create_argparser():
    defaults = dict(
        data_dir=opts.data_dir,
        test_dir=opts.test_dir,
        dire_mode=opts.dire_mode,
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=16,
        log_interval=200,
        save_interval=500,
        test_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
if __name__ == "__main__":
    run=WandbLogger()
    main(run)
