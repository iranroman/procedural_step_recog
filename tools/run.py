import argparse
import torch
import sys
from torch.utils.data import DataLoader
from step_recog.config.defaults import get_cfg
from step_recog import train, evaluate
from step_recog.datasets import Milly_multifeature

def parse_args():
    """
    Parse the following arguments for the video sliding pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Optional arguments",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    return cfg


def main():
    """
    Main function to spawn the process.
    """
    args = parse_args()
    cfg = load_config(args)

    # build the dataset
    tr_dataset = Milly_multifeature(cfg, split='train')
    vl_dataset = Milly_multifeature(cfg, split='validation')

    if cfg.TRAIN.ENABLE:
        tr_data_loader = DataLoader(
                tr_dataset, 
                shuffle=True, 
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=True) 
        vl_data_loader = DataLoader(
                vl_dataset, 
                shuffle=False, 
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=True) 
        model = train(tr_data_loader, 
                vl_data_loader,
                learn_rate = cfg.TRAIN.LR, 
                hidden_dim = cfg.MODEL.HIDDEN_SIZE, 
                EPOCHS = cfg.TRAIN.EPOCHS, 
                output_dim = cfg.MODEL.OUTPUT_DIM)
    if cfg.EVAL.ENABLE:
        print('loading best model')
        model.load_state_dict('model_best.pt')
        evaluate(vl_data_loader)


if __name__ == "__main__":
    main()
