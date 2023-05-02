import os 
import torch
import tqdm
import numpy as np
import pandas as pd

class Milly_multifeature(torch.utils.data.Dataset):

    def __init__(self, cfg, split='train'):

        self.data_path = cfg.DATASET.LOCATION
        self.data_path_audio = cfg.DATASET.AUDIO_LOCATION
        self.video_layer = cfg.DATASET.VIDEO_LAYER
        if split!='test':
            self.annotations_file = cfg.DATASET.TR_ANNOTATIONS_FILE
        elif split=='test':
            self.annotations_file = cfg.DATASET.VL_ANNOTATIONS_FILE
        self.context_length = cfg.MODEL.CONTEXT_LENGTH
        self.slide_hop_size = cfg.DATASET.HOP_SIZE
        self.video_fps = cfg.DATASET.FPS
        if split == 'train':
            self.image_augs = cfg.DATASET.INCLUDE_IMAGE_AUGMENTATIONS
            self.time_augs = cfg.DATASET.INCLUDE_TIME_AUGMENTATIONS
        else:
            self.image_augs = False
            self.time_augs = False
        self.rng = np.random.default_rng(12345)
        self._construct_loader(split)

    def _construct_loader(self, split):
	
        #frame_step_counter = {i:0 for i in range(8)}
        self.annotations = pd.read_csv(self.annotations_file,usecols=['video_id','start_frame','stop_frame','narration','verb_class'])
        self.datapoints = {}
        ipoint = 0
        video_ids = sorted(list(set(self.annotations.video_id)))
        rand_idx_ = self.rng.choice(len(video_ids),len(video_ids),replace=False)
        if split == 'train':
            video_ids = [video_ids[i] for i in rand_idx_[:int(len(video_ids)*0.85)]]
        elif split == 'validation':
            video_ids = [video_ids[i] for i in rand_idx_[int(len(video_ids)*0.85):]]
        self.max_nframes = 0
        for v in tqdm.tqdm(video_ids, total=len(video_ids)):
            # get video annotations
            vid_ann = self.annotations[self.annotations.video_id==v]
            start_frames = vid_ann.start_frame[1:].to_numpy()
            stop_frames = vid_ann.stop_frame[:-1].to_numpy()
            steps_ = []
            eframe = -1
            for r in vid_ann.iterrows():
                # need to fix steps with overlapping frames (how does that even make sense?)
                sframe = r[1].start_frame if r[1].start_frame > eframe else eframe + 1
                eframe = r[1].stop_frame if r[1].stop_frame > sframe else sframe + 1
                ### class offset of 34, (class, start_frame, end_frame)
                steps_.append((sframe,eframe,r[1].verb_class - 34))
            # checking that steps progress over time
            eframe = -1
            for ik, vv in enumerate(steps_):
                assert vv[0] > eframe 
                eframe = vv[1]
                if eframe > self.max_nframes:
                    self.max_nframes = eframe
            self.datapoints[ipoint] = {
                    'video_id': v,
                    'steps_frames': steps_
                }
            ipoint += 1

    def __getitem__(self, index, time_augs = [15,7.5,3.75], win_sizes = [4,2,1]):
        if self.image_augs:
            iaug = self.rng.choice(range(-1,20),replace=False) # 20 possible augmentations
        else:
            iaug = -1
        drecord = self.datapoints[index]
        
        # buffer up filenames and labels
        taug = 0
        curr_frame = 120
        hop_size = 15 # in units of frames
        frames = []
        labels = []
        wins = []
        for step in drecord['steps_frames']:

            hop_size = time_augs[taug]
            win_size = win_sizes[taug]
            if self.time_augs:
                nexttaug = self.rng.choice(3,replace=False) # 4 possible augmentations
            else:
                nexttaug = 0
            # fill-in before
            while True:
                if curr_frame > time_augs[nexttaug]*int(step[0]/time_augs[nexttaug]):
                    curr_frame = time_augs[nexttaug]*int((curr_frame-hop_size)/time_augs[nexttaug])
                    curr_frame += time_augs[nexttaug]
                    break
                else:
                    frames.append(curr_frame)
                    labels.append(8) # hard-coded for now. SORRY!!!
                    wins.append(win_size)
                    curr_frame += hop_size
            taug = nexttaug
            hop_size = time_augs[taug]
            win_size = win_sizes[taug]
            if self.time_augs:
                nexttaug = self.rng.choice(3,replace=False) # 4 possible augmentations
            else:
                nexttaug = 0
            # fill-in the step
            while True:
                if curr_frame > time_augs[nexttaug]*int(step[1]/time_augs[nexttaug]):
                    curr_frame = time_augs[nexttaug]*int((curr_frame-hop_size)/time_augs[nexttaug])
                    curr_frame += time_augs[nexttaug]
                    break
                else:
                    frames.append(curr_frame)
                    labels.append(step[-1])
                    wins.append(win_size)
                    curr_frame += hop_size
            taug = nexttaug
        ### debugging print
        #curr_l = 8
        #for f,l,w in zip(frames,labels,wins):
        #    if l == curr_l:
        #        print(f,l,w)
        #    else:
        #        print()
        #        curr_l = l
        #        print(f,l,w)
        #input()

        # now generate the paths to files to be loaded
        if iaug > -1:
            omni_paths = ['{}/{}_aug{}_{}secwin/{}/frame_{:010d}.npy'.format(self.data_path,drecord['video_id'],iaug,w,self.video_layer,int(f)) for f,w in zip(frames,wins)]
            obj_paths = ['/vast/bs3639/BBN/aug_yolo/{}_aug{}/frame_{:010d}.npz'.format(drecord['video_id'],iaug,int(f)) for f in frames]
            frame_paths = ['/vast/bs3639/BBN/aug_clip/{}_aug{}/frame_{:010d}.npy'.format(drecord['video_id'],iaug,int(f)) for f in frames]
        else:
            omni_paths = ['{}/{}_{}secwin/{}/frame_{:010d}.npy'.format(self.data_path,drecord['video_id'],w,self.video_layer,int(f)) for f,w in zip(frames,wins)]
            obj_paths = ['/vast/bs3639/BBN/yolo/{}/frame_{:010d}.npz'.format(drecord['video_id'],int(f)) for f in frames]
            frame_paths = ['/vast/bs3639/BBN/clip/{}/frame_{:010d}.npy'.format(drecord['video_id'],int(f)) for f in frames]
        omni_embeddings = []
        obj_embeddings = []
        frame_embeddings = []
        for o,b,f in zip(omni_paths,obj_paths,frame_paths):
            omni_embeddings.append(np.load(o))
            obj_features = np.load(b)['features']
            obj_boxes = np.load(b)['boxes']
            obj_conf = np.load(b)['confs']
            if len(obj_features) > 0:
                obj_features = obj_features
                obj_boxes = obj_boxes
                obj_conf = obj_conf[:,np.newaxis]
                obj_features = np.concatenate((obj_features,obj_boxes,obj_conf),axis=1)
            else:
                print('filling empty object detections')
                obj_features = np.zeros((25,517))
            np.random.shuffle(obj_features)
            if len(obj_features) > 25: # hard-coded for now
                obj_features = obj_features[:25]
            else:
                obj_features = np.pad(obj_features,((0,25-len(obj_features)),(0,0)))
            obj_embeddings.append(obj_features)
            frame_embeddings.append(np.load(f))
        return torch.from_numpy(np.array(omni_embeddings)), torch.from_numpy(np.array(obj_embeddings)), torch.from_numpy(np.array(frame_embeddings)),

    def __len__(self):
        return len(self.datapoints)
