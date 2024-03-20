import argparse
import torch
import sys
import os
import time
from torch.utils.data import DataLoader
from step_recog.config import load_config
from step_recog import train, evaluate, build_model
from step_recog.datasets import Milly_multifeature_v4
from sklearn.model_selection import KFold
import pandas as pd, pdb, numpy as np

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
    parser.add_argument("-i", "--kfold_iter", help = "Run a specific kfold iteration", dest = "forced_iteration", required = False, default = None, type = int)
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             here 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    omni,objs,frame,audio,lengths,labels,labels_t,frame_idx,ids = zip(*data)
    nomni_feats = 0 if omni[0].shape[0] == 0 else omni[0].shape[1]
    nobj_feats = 0 if objs[0].shape[0] == 0 else objs[0].shape[2]
    nframe_feats = 0 if frame[0].shape[0] == 0 else frame[0].shape[2]
    naudio_feats = 0 if audio[0].shape[0] == 0 else audio[0].shape[1]
    max_length = max(lengths) 
    omni_new = []
    objs_new = []
    frame_new = []
    audio_new = []
    labels_new = []
    labels_t_new = []
    mask_new = []
    frame_idx_new = []

    for i in range(len(data)):
      omni_empty = torch.zeros((max_length,nomni_feats))
      if nomni_feats > 0 and omni[i].shape[0] > 0:
        omni_empty[:omni[i].shape[0],:] = omni[i]
      omni_new.append(omni_empty)

      objs_empty = torch.zeros((max_length,25,nobj_feats))
      if nobj_feats > 0 and objs[i].shape[0] > 0:
        objs_empty[:objs[i].shape[0],...] = objs[i]
      objs_new.append(objs_empty)

      frame_empty = torch.zeros((max_length,1,nframe_feats))
      if nframe_feats > 0 and frame[i].shape[0] > 0:
        frame_empty[:frame[i].shape[0],:] = frame[i]
      frame_new.append(frame_empty)

      audio_empty = torch.zeros((max_length,naudio_feats))
      if naudio_feats > 0 and audio[i].shape[0] > 0:
        audio_empty[:audio[i].shape[0],:] = audio[i]
      audio_new.append(audio_empty)        

      labels_empty = torch.zeros((max_length))
      if labels[i].shape[0] > 0 and labels[i].shape[0] > 0:
        labels_empty[:labels[i].shape[0]] = labels[i]
      labels_new.append(labels_empty)

      labels_t_empty = torch.zeros((max_length,2))
      if labels_t[i].shape[0] > 0 and labels_t[i].shape[0] > 0:
        labels_t_empty[:labels_t[i].shape[0]] = labels_t[i][...,-2:]
      labels_t_new.append(labels_t_empty)

      mask_empty = torch.zeros((max_length,1))
      mask_empty[:labels[i].shape[0]] = 1
      mask_new.append(mask_empty)

      # frame_idx_empty = torch.zeros((max_length))
      # if frame_idx[i].shape[0] > 0 and frame_idx[i].shape[0] > 0:
      #   frame_idx_empty[:frame_idx[i].shape[0]] = frame_idx[i]
      # frame_idx_new.append(frame_idx_empty)        

    omni_new = torch.stack(omni_new)
    objs_new = torch.stack(objs_new)
    frame_new = torch.stack(frame_new)
    audio_new = torch.stack(audio_new)
    labels_new = torch.stack(labels_new)
    labels_t_new = torch.stack(labels_t_new)
    mask_new = torch.stack(mask_new)
    frame_idx_new = torch.stack(frame_idx)

    return omni_new.float(), objs_new.float(), frame_new.float(), audio_new.float(), labels_new.long(), labels_t_new.float(), mask_new.long(), frame_idx_new.long(), ids


def main():
  """
  Main function to spawn the process.
  """
  args = parse_args()
  cfg = load_config(args)

  if cfg.DATALOADER.NUM_WORKERS > 0:
    torch.multiprocessing.set_start_method('spawn')

  # build the dataset
  model_name = None
  timeout = 0

  kfold = True
  grid_test = False
  testing = False

  if kfold:
    train_kfold(cfg, args)    
  else:
    if cfg.TRAIN.ENABLE:
      tr_dataset = Milly_multifeature_v4(cfg, split='train')
      vl_dataset = Milly_multifeature_v4(cfg, split='validation')
      tr_data_loader = DataLoader(
              tr_dataset, 
              shuffle=True, 
              batch_size=cfg.TRAIN.BATCH_SIZE,
              num_workers=cfg.DATALOADER.NUM_WORKERS,
              collate_fn=collate_fn,
              drop_last=True,
              timeout=timeout)
      vl_data_loader = DataLoader(
              vl_dataset, 
              shuffle=False, 
              batch_size=cfg.TRAIN.BATCH_SIZE,
              num_workers=cfg.DATALOADER.NUM_WORKERS,
              collate_fn=collate_fn,
              drop_last=False,
              timeout=timeout)
  
##    if grid_test:  
##      grid_search(tr_data_loader, vl_data_loader, cfg)
##    elif testing:  
##      test_data(tr_data_loader, cfg)
##      test_data(vl_data_loader, cfg)      
    else:
      start_time = time.time()
      model_name = train(tr_data_loader, vl_data_loader,cfg)   
      print("Training time: ", time.time() - start_time)

      del vl_data_loader
      del tr_data_loader
      del vl_dataset
      del tr_dataset

      if cfg.EVAL.ENABLE:
        ts_dataset = Milly_multifeature_v4(cfg, split='test')
        ts_data_loader = DataLoader(
                ts_dataset, 
                shuffle=False, 
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                collate_fn=collate_fn,
                drop_last=False,
                timeout=timeout)          

        print('Loading the best model to evaluate')
        model, _ = build_model(cfg)      
        weights = torch.load(cfg.MODEL.CHECKPOINT_FILE_PATH if model_name is None else model_name)
        model.load_state_dict(model.update_version(weights))

        start_time = time.time()

        evaluate(model,ts_data_loader,cfg)
          
        print("Evaluating time: ", time.time() - start_time)

def train_kfold(cfg, args, k = 10):
  timeout = 0
  all_iteractions = False
  kf_train_test = KFold(n_splits = k)
  kf_train_val  = KFold(n_splits = (k - 1))

  data   = pd.read_csv(cfg.DATASET.TR_ANNOTATIONS_FILE)
  videos = data.video_id.unique()
  main_path = cfg.OUTPUT.LOCATION

  for idx, (aux_idx, test_idx) in enumerate(kf_train_test.split(videos), 1):
    video_aux  = videos[aux_idx]
    video_test = videos[test_idx]

    for jdx, (train_idx, val_idx) in enumerate(kf_train_val.split(video_aux), 1):
      if all_iteractions or (((jdx == idx) or (idx == k and jdx == 1)) and (args.forced_iteration is None or idx == args.forced_iteration)):
        print("==================== CROSS VALIDATION t{} v{}====================".format(idx, jdx))
        video_train = video_aux[train_idx]
        video_val   = video_aux[val_idx]    
        
        tr_dataset = Milly_multifeature_v4(cfg, split='train', filter=video_train)
        vl_dataset = Milly_multifeature_v4(cfg, split='validation', filter=video_val)

        tr_data_loader = DataLoader(
                tr_dataset, 
                shuffle=True, 
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                collate_fn=collate_fn,
                drop_last=True,
                timeout=timeout)
        vl_data_loader = DataLoader(
                vl_dataset, 
                shuffle=False, 
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                collate_fn=collate_fn,
                drop_last=False,
                timeout=timeout)

        train_path = os.path.join(main_path, "kfold", "fold_t{}-v{}".format(idx, jdx) )
        val_path   = os.path.join(train_path, "validation" )
        test_path  = os.path.join(train_path, "test" )

        if not os.path.isdir(val_path):
          os.makedirs( val_path )
        if not os.path.isdir(test_path):  
          os.makedirs( test_path )

  #      pdb.set_trace()
        cfg.OUTPUT.LOCATION = train_path
        model_name = train(tr_data_loader, vl_data_loader, cfg)    

        del tr_data_loader
        del tr_dataset          

        model, _ = build_model(cfg)      
        weights  = torch.load(model_name)
        model.load_state_dict(model.update_version(weights))      

        cfg.OUTPUT.LOCATION = val_path
        evaluate(model, vl_data_loader, cfg)      

        del vl_data_loader
        del vl_dataset

        ts_dataset = Milly_multifeature_v4(cfg, split='test', filter=video_test)
        ts_data_loader = DataLoader(
                ts_dataset, 
                shuffle=False, 
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                collate_fn=collate_fn,
                drop_last=False,
                timeout=timeout)
        
        cfg.OUTPUT.LOCATION = test_path
        evaluate(model, ts_data_loader, cfg)

        del ts_data_loader
        del ts_dataset        

if __name__ == "__main__":
    main()
