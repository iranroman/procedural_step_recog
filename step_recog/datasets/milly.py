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
        curr_frame = 60
        hop_size = 15 # in units of frames
        frames = []
        labels = []
        wins = []
        for step in drecord['steps_frames']:

            if self.time_augs:
                taug = self.rng.choice(3,replace=False) # 20 possible augmentations
            else:
                taug = 0
            hop_size = time_augs[taug]
            win_size = win_sizes[taug]
            print(hop_size)
            # fill-in before
            while step[0] > curr_frame:
                frames.append(curr_frame)
                labels.append(8) # hard-coded for now. SORRY!!!
                wins.append(win_size)
                curr_frame += hop_size
            
            if self.time_augs:
                taug = self.rng.choice(3,replace=False) # 20 possible augmentations
            else:
                taug = 0
            hop_size = time_augs[taug]
            win_size = win_sizes[taug]
            # fill-in the step
            while step[1] > curr_frame:
                frames.append(curr_frame)
                labels.append(step[-1])
                wins.append(win_size)
                curr_frame += hop_size

        # now generate the paths to files to be loaded
        omni_paths = ['{}/{}_aug{}_{}secwin/{}/frame_{:010d}.npy'.format(self.data_path,drecord['video_id'],iaug,w,self.video_layer,int(f)) for f,w in zip(frames,wins)]
        if iaug > -1:
            obj_paths = ['/vast/bs3639/BBN/aug_yolo/{}_aug{}/frame_{:010d}.npz'.format(drecord['video_id'],iaug,int(f)) for f in frames]
            frame_paths = ['/vast/bs3639/BBN/aug_clip/{}_aug{}/frame_{:010d}.npy'.format(drecord['video_id'],iaug,int(f)) for f in frames]
        else:
            obj_paths = ['/vast/bs3639/BBN/yolo/{}/frame_{:010d}.npz'.format(drecord['video_id'],int(f)) for f in frames]
            frame_paths = ['/vast/bs3639/BBN/clip/{}_aug{}/frame_{:010d}.npy'.format(drecord['video_id'],int(f)) for f in frames]
        omni_embeddings = []
        obj_embeddings = []
        frame_embeddings = []
        for o,b,f in zip(omni_paths,obj_paths,frame_paths):
            omni_embeddings.append(np.load(o))
            obj_embeddings.append(np.load(b))
            frame_embeddings.append(np.load(f))




        

        #print(drecord)
        #input()
        #last_frame = drecord['frame']
        #first_frame = last_frame - self.context_length * self.video_fps * self.slide_hop_size
        #loading_frame = last_frame
        #frames = []
        #frames_audio = []
        #while loading_frame > first_frame:
        #    frame_feats_path = os.path.join(self.data_path,drecord['video_id'],self.video_layer,'frame_{:010d}.npy'.format(loading_frame))
        #    frame_feats_path_audio = os.path.join(self.data_path_audio,drecord['video_id'],self.video_layer,'frame_{:010d}.npy'.format(loading_frame))
        #    if os.path.exists(frame_feats_path):
        #        frames.append(np.load(frame_feats_path))
        #    else:
        #        print(f'data for frame {loading_frame} not found. Recycling frame features')
        #        frames.append(frames[-1])
        #    if os.path.exists(frame_feats_path_audio):
        #        frames_audio.append(np.load(frame_feats_path_audio))
        #    else:
        #        print(f'audio data for frame {loading_frame} in {drecord["video_id"]} not found. Recycling frame features')
        #        frames_audio.append(0.0001*np.random.randn(*np.load(os.path.join(self.data_path_audio,'R1-P00_00',self.video_layer,'frame_0000000060.npy')).shape))
        #    loading_frame -= int(self.video_fps * self.slide_hop_size) 
        #video_features = np.flip(np.vstack(frames),axis=0)
        #audio_features = np.flip(np.vstack(frames_audio),axis=0)
        #vid_steps = self.annotations[self.annotations.video_id==drecord['video_id']]
        #action_idx = sum((drecord['frame'] - vid_steps['stop_frame'])>0)
        #step_label = vid_steps.iloc[action_idx].verb_class
        #return (torch.from_numpy(np.ascontiguousarray(video_features)), torch.from_numpy(np.ascontiguousarray(audio_features))), step_label

    def __len__(self):
        return len(self.datapoints)
