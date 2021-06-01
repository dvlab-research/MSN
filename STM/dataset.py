import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
# import torchvision
from torch.utils import data

import glob

class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False, max_obj_num=11):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)
        # assert 1<0, _imset_f

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        # self.K = 11 if
        # self.K = 2 if '2016' in imset else 11
        self.K = max_obj_num
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            N_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
                N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                # print('a')
                N_masks[f] = 255
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info


class YTVOS_val(data.Dataset):
    def __init__(self, root, imset='valid'):
        self.root = root
        self.mask_dir = os.path.join(root, imset, 'Annotations')
        self.image_dir = os.path.join(root, imset, 'JPEGImages')

        self.videos = []
        self.num_frames = {}
        self.frame_ids = {}
        self.num_objects = {}
        self.shape = {}
        self.start_frame = {}
        max_obj_num = 0

        for vid in sorted(os.listdir(self.image_dir)):
            if vid == '.' or vid == '..':
                continue
            self.videos.append(vid)
            self.num_frames[vid] = len(glob.glob(os.path.join(self.image_dir, vid, '*.jpg')))
            self.frame_ids[vid] = []
            self.start_frame[vid] = {}
            cur_obj_num = 0
            for t, name in enumerate(sorted(os.listdir(os.path.join(self.image_dir, vid)))):
                frame_id = name.split('.')[0]
                self.frame_ids[vid].append(frame_id)

                mask_file = os.path.join(self.mask_dir, vid, frame_id + '.png')
                if os.path.exists(mask_file):
                    mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                    self.shape[vid] = np.shape(mask)
                    max_obj = np.max(mask)
                    for k in range(1, max_obj + 1):
                        if (k in mask) and (k not in self.start_frame[vid].keys()):
                            self.start_frame[vid][k] = t
                    if max_obj > cur_obj_num:
                        cur_obj_num = max_obj

            self.num_objects[vid] = cur_obj_num
            max_obj_num = max(max_obj_num, self.num_objects[vid])
        print(max_obj_num)

        self.K = 6
        self.single_object = False

    def __len__(self):
        return len(self.videos)

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['frame_ids'] = self.frame_ids[video]
        info['num_frames'] = self.num_frames[video]
        info['shape'] = self.shape[video]
        info['start_frame'] = self.start_frame[video]

        N_frames = np.empty((self.num_frames[video],) + self.shape[video] + (3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],) + self.shape[video], dtype=np.uint8)
        for t in range(self.num_frames[video]):
            f = int(self.frame_ids[video][t])
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            N_frames[t] = np.array(Image.open(img_file).convert('RGB')) / 255.
            mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))
            if os.path.exists(mask_file):
                N_masks[t] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            else:
                N_masks[t] = 255

        #if(len(info['start_frame'].keys()) == 0):
        #    print(video)
        #    print(self.frame_ids[video])
        #    assert False
        #print(info['start_frame'])
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info
