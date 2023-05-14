import os
import time
import logging
import argparse

from omegaconf import OmegaConf

#from vits_extend.train import train
from vits_extend.tf_train import train
#torch.backends.cudnn.benchmark = True
#from tensorflow.python.ops.numpy_ops import np_config

if __name__ == '__main__':
    #np_config.enable_numpy_behavior()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="configs/base.yaml",
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, default="sovits5.0",
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    assert hp.data.hop_length == 320, \
        'hp.data.hop_length must be equal to 320, got %d' % hp.data.hop_length
    train(0, args, args.checkpoint_path, hp, hp_str)
    # args.num_gpus = 0
    # torch.manual_seed(hp.train.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(hp.train.seed)
    #     args.num_gpus = torch.cuda.device_count()
    #     print('Batch size per GPU :', hp.train.batch_size)

    #     if args.num_gpus > 1:
    #         mp.spawn(train, nprocs=args.num_gpus,
    #                  args=(args, args.checkpoint_path, hp, hp_str,))
    #     else:
    #         train(0, args, args.checkpoint_path, hp, hp_str)
    # else:
    #     print('No GPU find!')
