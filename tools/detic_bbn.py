import collections
import os
import glob
import tqdm
import sys


# models_yolo = "/ext3/miniconda3/lib/python3.11/site-packages/ultralytics"
##models_yolo = "/scratch/ffd2011/yolov5-master"
# models_yolo = "/scratch/ffd2011/yolov7-main"
# sys.path.insert(0, models_yolo)

#import h5py
import cv2
import numpy as np
import torch
from torch import nn
#from ptgprocess.detic import Detic
from ptgprocess.yolo import BBNYolo
#from ptgprocess import box_util
import clip
from step_recog.full import ClipPatches
import pdb

import pathtrees as pt
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# VOCABS = {
#     'pinwheels': [
#         'cutting board', 'chopping board', 
#         'knife', 'butter knife', 
#         'tortilla',
#         'flour tortilla',
#         'flauta',
#         'burrito',
#         'wraps',
#         'circular flour tortilla',
#         'sliced wraps',
#         'cut wraps',
#         'Jar of peanut butter',
#         'Jar of nut butter',
#         'Jar of jelly',
#         'Jar of jam',
#         'paper towel',
#         'paper towel roll',
#         'paper towel sheet',
#         'napkin',
#         'toothpicks',
#         'dental floss',
#         'string',
#         'plate',
#     ],
#     'coffee': [
#         'water',
#         'pouring water'
#         'brown bag',
#         'brown bag of coffee beans',
#         'coffee beans',
#         'coffee',
#         'liquid measuring cup',
#         'measuring cup',
#         'electric kettle',
#         'hot water kettle'
#         'electronic scale',
#         'kitchen scale',
#         'coffee grinder',
#         'filter cone dripper (stainless steel)',
#         'coffee cone dripper',
#         'brown paper coffee filter',
#         'paper filter',
#         'paper cone',
#         'coffee mug',
#         'thermometer',
#     ],
#     'mugcake': [
#         'white bag of flour',
#         'bag of flour',
#         'sugar container',
#         'Domino sugar container',
#         'baking powder container',
#         'baking powder',
#         'iodized salt container',
#         'iodized salt',
#         'bottle of canola oil',
#         'water',
#         'sink',
#         'faucet',
#         'vanilla extract bottle',
#         'vanilla extract box',
#         'Container of chocolate frosting (premade)',
#         'measuring spoons',
#         'mixing bowl',
#         'whisk',
#         'paper cupcake liner',
#         'coffee mug',
#         'plate',
#         'microwave',
#         'zip-top bag',
#         'ziploc bag',
#         'box of ziploc bags',
#         'scissors',
#         'spoon',
#         'toothpicks',
#     ]
# }
# MILLY_KEYS = {'R0': 'pinwheels', 'R1': 'coffee', 'R2': 'mugcake'}

class Extractor:
    ROOT = '/vast/{user}/BBN'
    RGB_FRAMES = '{out_format}/{video_id}/frame_{frame_id}.{ext}'
    AUG_FRAMES = '{out_format}/{video_id}_aug{aug_id}/frame_{frame_id}.{ext}'
    def __init__(self, **kw):
        print("Using skill:", kw.get('skill'))
        #self.detic = Detic(vocab=['cat'], **kw)   # random vocab for faster init
        self.yolo = BBNYolo(**kw)
        self.encoder = ClipPatches().to(device)

    def run(self, frame_dir):#, out_dir=DEFAULT_OUTPUT_DIR):
        # loop over participants
        for d in tqdm.tqdm(glob.glob(os.path.join(frame_dir, '*/rgb_frames/*'))):
            self._run_one(d, out_dir)

    def run_one(self, frame_dir, out_root=None):#, out_dir=DEFAULT_OUTPUT_DIR, vocab=None
        #if not vocab:
        #    vocab_key = frame_dir.split('/')[-1].split('-')[0]
        #    vocab = VOCABS[MILLY_KEYS.get(vocab_key, vocab_key)]
        #self.detic.set_vocab(vocab)

        tqdm.tqdm.write(f'{frame_dir}...')
        print(self, frame_dir, out_root)
        rel_pattern = pt.Path(self.AUG_FRAMES if 'aug' in frame_dir else self.RGB_FRAMES)

        _root = pt.Path(frame_dir).parent.parent
        pattern = _root / rel_pattern
        out_pattern = pt.Path(out_root or _root) / rel_pattern
        print(pattern, out_pattern)

        prefix = 'aug_' if 'aug' in frame_dir else ''
        
        user = os.getenv('USER')
        assert user

        vid_name = os.path.basename(frame_dir)
        d = pattern.parent.parse(frame_dir)
##        d.pop('aug_id', None)
        fs = pattern.specify(**d).glob()
        print(len(fs), fs[:2])

        pbar = tqdm.tqdm(fs)
        for f in pbar:
            clip_frame_fname = self.get_out_fname(pattern, out_pattern, f, user=user, out_format=f'{prefix}clip', ext='npy')
            clip_box_fname = self.get_out_fname(pattern, out_pattern, f, user=user, out_format=f'{prefix}yolo', ext='npz')
            detic_box_fname = None #self.get_out_fname(pattern, out_pattern, f, user=user, out_format=f'{prefix}detic', ext='npz')
            if not any((clip_frame_fname, clip_box_fname, detic_box_fname)):
                continue
            pbar.set_description(f)
            
            # load image
            im = cv2.imread(f)
            # compute
            if clip_frame_fname:
                self.compute_store_clip_frame(im, clip_frame_fname)
            if clip_box_fname:
                self.compute_store_clip_boxes(im, clip_box_fname)
            if detic_box_fname:
                self.compute_store_detic(im, detic_box_fname)

    def get_out_fname(self, pattern, out_pattern, fname, **replacement):
        out_fname = out_pattern.format(**{**pattern.parse(fname), **replacement})
        if os.path.exists(out_fname):
            return 
        if not os.path.exists(os.path.dirname(out_fname)):
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        return out_fname

    def compute_store_clip_frame(self, im, out_fname):
        output = self.encoder(im, include_frame = True).cpu()
        np.save(out_fname, output)

    def compute_store_clip_boxes(self, im, out_fname):
        output = self.yolo(im)
        xywh, *_ = self.yolo.unpack_results(output, 'xywh')
        xyxyn, _, class_ids, labels, confs, _ = self.yolo.unpack_results(output)
        #xywh = output[0].cpu().numpy()
        #xywh = box_util.tlbr2tlwh(box_util.unnorm(xyxyn, im.shape))
        features = self.encoder(im, xywh).cpu()
        np.savez(out_fname,
            boxes=xyxyn,
            class_ids=class_ids,
            labels=labels,
            features=features,
            confs=confs)

    def compute_store_detic(self, im, out_fname):
        output = self.model(im)
        insts = output["instances"].to("cpu")
        np.savez(out_fname,
            boxes=insts.pred_boxes.tensor.numpy(),
            class_ids=insts.pred_classes.numpy().astype(int),
            confs=insts.scores.numpy(),
            box_confs=insts.box_scores.numpy(),
            clip_features=insts.clip_features.numpy())

if __name__ == '__main__':
    import fire
    with torch.no_grad():
        fire.Fire(Extractor)

