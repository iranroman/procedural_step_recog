import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.utils import save_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

__DATA_DIR__ = '/vast/irr2020/BBN'

class ProceduralStepsDataset(Dataset):
    """
    A dataset class for procedural steps that supports k-fold cross-validation and data partitioning.

    Attributes:
        data_dir (str): Directory where the data files are stored.
        split (str): Specifies if the dataset is for 'training' or 'testing'.
        k_fold_cross_validation (bool): If True, enables k-fold cross-validation.
        Kfolds (int): The number of folds for cross-validation.
        fold_split (str): Specifies the type of data partition ('train' or 'val').
        val_fold (int): Specifies the fold to be used as the validation set.
        video_names (list): List of video names after applying k-fold logic.
    """
    def __init__(self, data_dir, split='training', k_fold_cross_validation=True,
                 Kfolds=5, fold_split='train', val_fold=1, frame_buffer_size=32,
                 target_fps=10, time_augment=True, resolution=224,
                 target_fps_range=5, 
                 image_augmentations = ['RandomRotation','HorizontalFlip','RandomPerspective',
                     'RandomHue'],
                 augment_images = True,
                 img_aug_p = 0.85,
                 ):
        """
        Initializes the ProceduralStepsDataset with the provided parameters and prepares the dataset.
        """
        self.data_dir = data_dir
        self.video_names = []
        self.load_and_shuffle_video_names(split)
        self.partition_videos(k_fold_cross_validation, Kfolds, fold_split, val_fold)
        self.video_info = self.load_video_info(split)
        self.split = split
        self.frame_buffer_size = frame_buffer_size
        self.time_augment = time_augment
        self.target_fps = target_fps
        if self.time_augment:
            self.min_target_fps = target_fps - target_fps_range
            self.max_target_fps = target_fps + target_fps_range
        self.image_resolution = resolution
        self.image_augmentations = image_augmentations
        self.augment_images = augment_images
        self.img_aug_p = img_aug_p
        path_to_step_descriptions = os.path.join(self.data_dir,'procedure_descriptions')
        all_description_files = os.listdir(path_to_step_descriptions)
        norm_desc_files = [f for f in all_description_files if 'norm' in f]
        desc_files = [f for f in all_description_files if 'norm' not in f]
        self.steps2normsteps = {}
        for f,fn in zip(desc_files,norm_desc_files):
            with open(os.path.join(path_to_step_descriptions,fn), 'r') as file:
                norm_file = file.read().splitlines()
            with open(os.path.join(path_to_step_descriptions,f), 'r') as file:
                file = file.read().splitlines()
            self.steps2normsteps.update({step[3:]:nstep[3:] for step,nstep in zip(file[2:],norm_file[2:])})
        self.normsteps2index = {v:i for i,v in enumerate(sorted(list(set(self.steps2normsteps.values()))))}

    def load_and_shuffle_video_names(self, split):
        """
        Loads and shuffles video names from a file.
        """
        path = os.path.join(self.data_dir, f'{split}_video_names_shuffled.txt')
        try:
            with open(path, 'r') as file:
                self.shuffled_video_names = file.read().splitlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {path} does not exist")

    def partition_videos(self, k_fold_cross_validation, Kfolds, fold_split, val_fold):
        """
        Partitions video names into training and validation sets based on k-fold cross-validation settings.
        """
        if k_fold_cross_validation:
            assert 1 <= val_fold <= Kfolds, "val_fold must be within the range of Kfolds"
            nvideos = len(self.shuffled_video_names)
            k, m = divmod(nvideos, Kfolds)
            self.video_folds = [self.shuffled_video_names[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(Kfolds)]
            if fold_split == 'train':
                self.video_names = [name for i, fold in enumerate(self.video_folds) if i + 1 != val_fold for name in fold]
            elif fold_split == 'val':
                self.video_names = self.video_folds[val_fold - 1]
        else:
            self.video_names = self.shuffled_video_names

    def load_video_info(self, split):
        """
        Loads video information from a CSV file.
        """
        path = os.path.join(self.data_dir, f'{split}_video_info.csv')
        try:
            video_info = pd.read_csv(path).set_index('video_id')
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {path} does not exist")
        return video_info

    def __len__(self):
        """
        Returns the number of videos in the dataset.
        """
        return len(self.video_names)

    def __getitem__(self, idx):
        """
        Retrieves the video at the specified index.
        """
        video_name = self.video_names[idx]
        print(video_name)
        video_fps = self.video_info.loc[video_name].video_fps
        video_dur = self.video_info.loc[video_name].video_duration
        video_frames_path = os.path.join(self.data_dir,f'{self.split}_videos',video_name,'frames')
        path = os.path.join(self.data_dir, f'{self.split}_videos',video_name,f'{video_name}.skill_labels_by_frame.txt')
        with open(path, 'r') as file:
            video_labels = file.read().splitlines()
        video_labels = [l.split('\t') for l in video_labels]
        video_framestamps = np.array([l[:2] for l in video_labels]).astype(int)
        video_step_names = [l[-1] for l in video_labels]
        assert np.array_equal(video_framestamps, video_framestamps[np.argsort(video_framestamps[:,0])])
        ranges = []
        for i in video_framestamps:
            ranges.append(list(range(i[0],i[1]+1)))
        all_used = [i for i in ranges[-1]]
        for i in reversed(range(len(ranges)-1)):
            ranges[i] = sorted(list(set(ranges[i]) - set(all_used)))
            all_used.extend(ranges[i])


        image_augment = np.random.choice([True, False], p=[self.img_aug_p,(1-self.img_aug_p)])

        labels = []
        step_label = []

        if 'RandomRotation' in self.image_augmentations:
            curr_rotation = 0
        if 'RandomPerspective' in self.image_augmentations:
            xtopl = 0 
            xbotl = 0 
            ytopl = 0
            ytopr = 0 
            xtopr = 0 
            xbotr = 0 
            ybotl = 0 
            ybotr = 0 
        if 'HorizontalFlip' in self.image_augmentations:
            flip = np.random.choice([True, False])
        if 'RandomHue' in self.image_augmentations:
            curr_hue = 0.0

        # create the initial buffer of frames
        first_frame = read_image(os.path.join(video_frames_path,f'frame_{1:010d}.jpg'))
        if self.augment_images and image_augment:
            if 'HorizontalFlip' in self.image_augmentations:
                if flip:
                    first_frame = torch.flip(first_frame,dims=(-1,))
        width_index = (first_frame.shape[-1] - first_frame.shape[-2])//2 # assuming W>H
        loader_frame_count = 1
        video_frames = [first_frame[:,:,width_index:width_index+self.image_resolution]]
        #os.makedirs(video_name)
        #save_image(video_frames[-1]/255,f'{video_name}/frame{loader_frame_count}.png')
        for i in range(self.frame_buffer_size-1):
            loader_frame_count += 1
            
            if self.augment_images and image_augment:
                width_index += np.random.choice([-1,1])
                width_index = np.clip(width_index,0,first_frame.shape[-1]-self.image_resolution-1)
                if 'RandomRotation' in self.image_augmentations:
                    curr_frame = T.functional.rotate(first_frame,curr_rotation)
                    curr_rotation += np.random.choice([-1,1])
                    curr_rotation = int(np.clip(curr_rotation,-180,180))
            else:
                curr_frame = first_frame

            video_frames.append(curr_frame[:,:,width_index:width_index+self.image_resolution])
            #save_image(video_frames[-1]/255,f'{video_name}/frame{loader_frame_count}.png')

        # get first label
        frame_label = [i for i,r in enumerate(ranges) if 1 in r]
        assert len(frame_label) <= 1
        if frame_label:
            labels.append(frame_label[0])
            step_label.append(self.normsteps2index[self.steps2normsteps[video_step_names[labels[-1]]]])
        else:
            labels.append(-1)
            step_label.append(len(self.normsteps2index))

        buffer_time = 0

        while buffer_time < video_dur-1:
            loader_frame_count += 1

            # dequeue and add frames as we progress through the video
            buffer_time += 1/self.target_fps

            frame_number = int(video_fps*buffer_time)
            curr_frame = read_image(os.path.join(video_frames_path,f'frame_{frame_number:010d}.jpg'))
            frame_label = [i for i,r in enumerate(ranges) if frame_number in r]
            assert len(frame_label) <= 1
            if frame_label:
                labels.append(frame_label[0])
                step_label.append(self.normsteps2index[self.steps2normsteps[video_step_names[labels[-1]]]])
            else:
                labels.append(-1)
                step_label.append(len(self.normsteps2index))
            if self.augment_images and image_augment:
                edge_size = 50
                top_indices = np.random.choice(range(curr_frame.shape[-2]),edge_size,replace=False)
                bot_indices = np.random.choice(range(curr_frame.shape[-2]),edge_size,replace=False)
                top_indices.sort()
                bot_indices.sort()
                curr_frame = torch.cat((curr_frame[:,top_indices[::-1].copy(),:],curr_frame,curr_frame[:,bot_indices[::-1].copy(),:]),dim=-2)
                #left_indices = np.random.choice(range(curr_frame.shape[-1]),edge_size,replace=False)
                #right_indices = np.random.choice(range(curr_frame.shape[-1]),edge_size,replace=False)
                #left_indices.sort()
                #right_indices.sort()
                #curr_frame = torch.cat((curr_frame[:,:,left_indices[::-1].copy()],curr_frame,curr_frame[:,:,right_indices[::-1].copy()]),dim=-1)
                width_index += np.random.choice([-1,1])
                width_index = np.clip(width_index,0,first_frame.shape[-1]-self.image_resolution-1)
                if 'RandomPerspective' in self.image_augmentations:
                    curr_frame = T.functional.perspective(curr_frame,
                            [
                                [0,0],
                                [curr_frame.shape[-1],0],
                                [0,curr_frame.shape[-2]],
                                [curr_frame.shape[-1],curr_frame.shape[-2]],
                            ],
                            [
                                [xtopl,ytopl],
                                [curr_frame.shape[-1]+xtopr,0+ytopr],
                                [xbotl,curr_frame.shape[-2]+ybotl],
                                [curr_frame.shape[-1]+xbotr,curr_frame.shape[-2]+ybotr],
                            ],
                    )
                    xtopl += int(np.random.choice([-3,3]))
                    ytopl += int(np.random.choice([-3,3]))
                    xtopr += int(np.random.choice([-3,3]))
                    ytopr += int(np.random.choice([-3,3]))
                    xbotl += int(np.random.choice([-3,3]))
                    ybotl += int(np.random.choice([-3,3]))
                    xbotr += int(np.random.choice([-3,3]))
                    ybotr += int(np.random.choice([-3,3]))
                    xtopl = int(np.clip(xtopl,-10,edge_size/2))
                    xbotl = int(np.clip(xbotl,-10,edge_size/2))
                    ytopl = int(np.clip(ytopl,-10,edge_size/2))
                    ytopr = int(np.clip(ytopr,-10,edge_size/2))
                    xtopr = int(np.clip(xtopr,-edge_size/2,10))
                    xbotr = int(np.clip(xbotr,-edge_size/2,10))
                    ybotl = int(np.clip(ybotl,-edge_size/2,10))
                    ybotr = int(np.clip(ybotr,-edge_size/2,10))
                if 'RandomRotation' in self.image_augmentations:
                    curr_frame = T.functional.rotate(curr_frame,curr_rotation)
                    curr_rotation += np.random.choice([-1,1])
                    curr_rotation = int(np.clip(curr_rotation,-180,180))
                if 'HorizontalFlip' in self.image_augmentations:
                    if flip:
                        curr_frame = torch.flip(curr_frame,dims=(-1,))
                if 'RandomHue' in self.image_augmentations:
                    curr_frame = T.functional.adjust_hue(curr_frame,curr_hue)
                    curr_hue += np.random.choice([-0.01,0.01])
                    curr_hue = float(np.clip(curr_hue,-0.25,0.25))
                curr_frame = curr_frame[:,edge_size:-edge_size,:]

           
            #pil_img = TF.to_pil_image(curr_frame.byte())
            #draw = ImageDraw.Draw(pil_img)
            #text = f"fps = {self.target_fps} (frame {frame_number})"
            #font = ImageFont.load_default()#truetype("arial.ttf", size=24)
            #draw.text((width_index+10,5), text, font=font, fill="white") 
            #if labels[-1] != -1:
            #    text = f"step class {step_label[-1]} ({self.steps2normsteps[video_step_names[labels[-1]]]})"
            #else:
            #    text = f"step class {step_label[-1]} ('no step')"
            #draw.text((width_index+10,15), text, font=font, fill="white") 
            #if labels[-1] != -1:
            #    text = f"detail: {video_step_names[labels[-1]]})"
            #else:
            #    text = f"detail: 'no step'"
            #draw.text((width_index+10,25), text, font=font, fill="white") 
            #text = f"width_index = {width_index}"
            #font = ImageFont.load_default()#truetype("arial.ttf", size=24)
            #draw.text((width_index+10,35), text, font=font, fill="white") 
            #curr_frame = TF.to_tensor(pil_img)*255

            video_frames.append(curr_frame[:,:,width_index:width_index+self.image_resolution])
            #save_image(video_frames[-1]/255,f'{video_name}/frame{loader_frame_count}.png')

            # new update target_fps
            if self.time_augment:
                self.target_fps += np.random.choice([-1,1],p=[0.45,0.55])
                self.target_fps = int(np.clip(self.target_fps,self.min_target_fps,self.max_target_fps))

        state_machine = torch.zeros(len(step_label),len(self.normsteps2index)) # unobserved, current, done
        for i in set(step_label):
            if i != 18:
                first_index = step_label.index(i)
                last_index = len(step_label) - 1 - step_label[::-1].index(i)
                state_machine[first_index:last_index+1,i] = 1
                state_machine[last_index+1:,i] = 2

        video_frames = torch.stack(video_frames)/255
        step_label = torch.tensor(step_label)

        return video_frames, step_label, state_machine.long()





if __name__ == "__main__":


    tr_dataset = ProceduralStepsDataset(__DATA_DIR__,fold_split='train')
    vl_dataset = ProceduralStepsDataset(__DATA_DIR__,fold_split='val',augment_images=False,time_augment=False)

    for i in np.random.choice(range(len(tr_dataset)),len(tr_dataset),replace=False):
        tr_dataset[i]
    for i in np.random.choice(range(len(vl_dataset)),len(vl_dataset),replace=False):
        vl_dataset[i]
