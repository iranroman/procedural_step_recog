import os 
import torch
import tqdm
import numpy as np
import pandas as pd

class Milly_multifeature(torch.utils.data.Dataset):

    def __init__(self, cfg, split='train'):

        self.data_path = cfg.DATASET.LOCATION
        self.video_layer = cfg.DATASET.VIDEO_LAYER
        if split=='train':
            self.annotations_file = cfg.DATASET.TR_ANNOTATIONS_FILE
        elif split=='validation':
            self.annotations_file = cfg.DATASET.VL_ANNOTATIONS_FILE
        self.context_length = cfg.MODEL.CONTEXT_LENGTH
        self.slide_hop_size = cfg.DATASET.HOP_SIZE
        self.video_fps = cfg.DATASET.FPS
        self._construct_loader()

    def _construct_loader(self):
	
        self.annotations = pd.read_csv(self.annotations_file,usecols=['video_id','start_frame','stop_frame','narration','verb_class'])
        self.datapoints = {}
        ipoint = 0
        #video_ids = os.listdir(self.data_path)
        video_ids = sorted(list(set(self.annotations.video_id)))
        # exclude some faulty videos
        video_ids = [v for v in video_ids if v not in ['R1-P10_06','R2-P04_08']]
        for v in tqdm.tqdm(video_ids, total=len(video_ids)):
            vid_last_step_frame = self.annotations[self.annotations.video_id==drecord['video_id']].iloc[-1].stop_frame
            print(type(vid_last_step_frame))
            vid_features = sorted(os.listdir(os.path.join(self.data_path,v,self.video_layer)))
            for f in vid_features:
                curr_frame = int(f.split('_')[1][:-4])
                if curr_frame < vid_last_step_frame:
                    self.datapoints[ipoint] = {
                                'video_id':v,
                                'npy_file':f,
                                'frame': curr_frame
                            }
                    ipoint += 1

    def __getitem__(self, index):
        drecord = self.datapoints[index]
        last_frame = drecord['frame']
        first_frame = last_frame - self.context_length * self.video_fps * self.slide_hop_size
        loading_frame = last_frame
        frames = []
        while loading_frame > first_frame:
            frame_feats_path = os.path.join(self.data_path,drecord['video_id'],self.video_layer,'frame_{:010d}.npy'.format(loading_frame))
            if os.path.exists(frame_feats_path):
                frames.append(np.load(frame_feats_path))
            else:
                Warning(f'data for frame {loading_frame} not found. Recycling frame features')
                frames.append(frames[-1])
            loading_frame -= int(self.video_fps * self.slide_hop_size) 
        video_features = np.flip(np.vstack(frames),axis=0)
        vid_steps = self.annotations[self.annotations.video_id==drecord['video_id']]
        action_idx = sum((drecord['frame'] - vid_steps['stop_frame'])>0)
        step_label = vid_steps.iloc[action_idx].verb_class
        return torch.from_numpy(np.ascontiguousarray(video_features)),step_label

    def __len__(self):
        return len(self.datapoints)
