import os
import gc
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from carla_utils.route_utils import command_to_branch_id


class CarlaImageCache:
    def __init__(self, dataframe, img_size):
        self.data = dataframe
        print(f"Loading {len(self.data)} images into RAM (uint8)...")
        self.images = {}
        for idx in tqdm(range(len(self.data))):
            img_name = self.data.iloc[idx]['image_path']
            img_path = os.path.join(img_name) 
            try:
                with Image.open(img_path) as img:
                    self.images[idx] = np.array(img, dtype=np.uint8)
            except:
                print(f"Warning: Failed to load {img_path}. Using black image.")
                self.images[idx] = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        gc.collect()

class DiskCarlaDataset(Dataset):
    def __init__(self, dataframe, img_dir, indices, is_train=True):
        """
        Reads images from disk on-the-fly.
        """
        self.data = dataframe
        self.img_dir = img_dir
        self.indices = indices
        self.is_train = is_train
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        current_traj = self.data.iloc[idx]['trajectory_id']
        
        do_flip = self.is_train and (random.random() < 0.5)
        
        frames = []
        for t in range(1):
            target_idx = idx - t
            
            # Check continuity
            if target_idx < 0 or self.data.iloc[target_idx]['trajectory_id'] != current_traj:
                # Boundary: duplicate the last valid frame
                arr = frames[0] if frames else self._load_image(idx, do_flip)
                frames.insert(0, arr)
            else:
                frames.insert(0, self._load_image(target_idx, do_flip))

        # Stack numpy arrays: [4, H, W, 3] -> [12, H, W]
        img_stack = np.concatenate(frames, axis=2) 
        img_stack = np.transpose(img_stack, (2, 0, 1)) 
        
        text = self.encoded_instructions[idx]
        
        # --- TARGETS & SCALING ---
        row = self.data.iloc[idx]
        steer = row['steer']
        
        # 1. Flip Logic
        if do_flip: steer = -steer
        
        # 2. Amplification Logic
        steer = steer * CONTROL_SCALE

        throttle = row['throttle']
        brake = row['brake']
        speed = row['speed']
        accel = throttle - brake
        accel = accel * CONTROL_SCALE
        
        target_action = torch.tensor([steer, accel], dtype=torch.float32)
        
        if idx > 0 and self.data.iloc[idx-1]['trajectory_id'] == current_traj:
            prev_row = self.data.iloc[idx-1]
            p_steer = prev_row['steer']
            if do_flip: p_steer = -p_steer
            p_steer = p_steer * CONTROL_SCALE
            p_throttle = prev_row['throttle']
            p_brake = prev_row['brake']
            p_speed = prev_row['speed']
            p_accel = p_throttle - p_brake
            p_accel = p_accel * CONTROL_SCALE
            
            prev_action = torch.tensor([p_steer, p_accel, p_speed], dtype=torch.float32)
        else:
            prev_action = torch.zeros(3, dtype=torch.float32)
            
        return img_stack, prev_action, target_action

    def _load_image(self, idx, do_flip):
        img_name = self.data.iloc[idx]['image_path']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Use PIL for loading
        try:
            with Image.open(img_path) as img:
                arr = np.array(img, dtype=np.uint8)
        except:
            # Fallback black image
            arr = np.zeros((256, 256, 3), dtype=np.uint8)
            
        if do_flip:
            arr = np.fliplr(arr).copy()
            
        return arr

class CarlaFastDataset(Dataset):
    def __init__(self, cache_manager, indices, flip_prob=0.5, is_train=True):
        self.manager = cache_manager
        self.indices = indices
        self.is_train = is_train
        self.FLIP_PROB = flip_prob

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        current_traj = self.manager.data.iloc[idx]['trajectory_id']
        instruction = self.manager.data.iloc[idx]['instruction']

        is_symmetric = "follow" in instruction or "change" in instruction
        do_flip = self.is_train and is_symmetric and (random.random() < self.FLIP_PROB)
        
        frames = []
        frames.insert(0, self._get_raw(idx, do_flip))

        img_stack = np.concatenate(frames, axis=2) 
        img_stack = np.transpose(img_stack, (2, 0, 1)) 

        if do_flip:
            if "left" in instruction:
                instruction = instruction.replace("left", "right")
            elif "right" in instruction:
                instruction = instruction.replace("right", "left")
        branch_id = command_to_branch_id(instruction)

        row = self.manager.data.iloc[idx]
        steer = row['steer']
        if do_flip: steer = -steer

        throttle = row['throttle']
        brake = row['brake']
        speed = row['speed']

        if speed > 2.0 and throttle < 0.01 and brake < 0.01:
            throttle = 0.2

        accel = throttle - brake
        accel = accel
        target_action = torch.tensor([steer, accel], dtype=torch.float32)
        
        # Previous Action (Apply same cleaning)
        if idx > 0 and self.manager.data.iloc[idx-1]['trajectory_id'] == current_traj:
            prev_row = self.manager.data.iloc[idx-1]
            p_steer = prev_row['steer']
            if do_flip: p_steer = -p_steer

            p_throttle = prev_row['throttle']
            p_brake = prev_row['brake']
            p_speed = prev_row['speed']
            p_accel = p_throttle - p_brake
            p_accel = p_accel
            
            prev_action = torch.tensor([p_steer, p_accel, p_speed], dtype=torch.float32)
        else:
            prev_action = torch.zeros(3, dtype=torch.float32)
            
        return img_stack, branch_id, prev_action, target_action

    def _get_raw(self, idx, do_flip):
        arr = self.manager.images[idx] 
        if do_flip: arr = np.fliplr(arr).copy()
        else: arr = arr.copy()
        return arr

class CarlaFastDatasetGated(Dataset):
    def __init__(self, cache_manager, indices, flip_prob=0.5, is_train=True):
        self.manager = cache_manager
        self.indices = indices
        self.is_train = is_train
        self.FLIP_PROB = flip_prob

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        current_traj = self.manager.data.iloc[idx]['trajectory_id']
        instruction = self.manager.data.iloc[idx]['instruction']
        is_rainy = self.manager.data.iloc[idx]['is_rainy']
        is_crowded = self.manager.data.iloc[idx]['is_crowded']

        is_symmetric = "follow" in instruction or "change" in instruction
        do_flip = self.is_train and is_symmetric and (random.random() < self.FLIP_PROB)
        
        frames = []
        frames.insert(0, self._get_raw(idx, do_flip))

        img_stack = np.concatenate(frames, axis=2) 
        img_stack = np.transpose(img_stack, (2, 0, 1)) 

        if do_flip:
            if "left" in instruction:
                instruction = instruction.replace("left", "right")
            elif "right" in instruction:
                instruction = instruction.replace("right", "left")
        branch_id = command_to_branch_id(instruction)

        task_vec = torch.zeros(6, dtype=torch.float32)
        task_vec[branch_id] = 1.0
        '''
        if is_rainy:
            task_vec[6] = 1.0
        if is_crowded:
            task_vec[7] = 1.0
        '''

        row = self.manager.data.iloc[idx]
        steer = row['steer']
        if do_flip: steer = -steer

        throttle = row['throttle']
        brake = row['brake']
        speed = row['speed']

        if speed > 2.0 and throttle < 0.01 and brake < 0.01:
            throttle = 0.2

        accel = throttle - brake
        accel = accel
        target_action = torch.tensor([steer, accel], dtype=torch.float32)
        
        # Previous Action (Apply same cleaning)
        if idx > 0 and self.manager.data.iloc[idx-1]['trajectory_id'] == current_traj:
            prev_row = self.manager.data.iloc[idx-1]
            p_steer = prev_row['steer']
            if do_flip: p_steer = -p_steer

            p_throttle = prev_row['throttle']
            p_brake = prev_row['brake']
            p_speed = prev_row['speed']
            p_accel = p_throttle - p_brake
            p_accel = p_accel
            
            prev_action = torch.tensor([p_steer, p_accel, p_speed], dtype=torch.float32)
        else:
            prev_action = torch.zeros(3, dtype=torch.float32)
            
        return img_stack, branch_id, task_vec, prev_action, target_action

    def _get_raw(self, idx, do_flip):
        arr = self.manager.images[idx] 
        if do_flip: arr = np.fliplr(arr).copy()
        else: arr = arr.copy()
        return arr


def collate_fn(batch):
    imgs, branch_ids, prevs, targets = zip(*batch)
    imgs = torch.from_numpy(np.stack(imgs)) 
    branch_ids = torch.tensor(branch_ids, dtype=torch.long)
    prevs = torch.stack(prevs)
    targets = torch.stack(targets)
    return imgs, branch_ids, prevs, targets

def collate_fn_gated(batch):
    imgs, branch_ids, task_vec, prevs, targets = zip(*batch)
    imgs = torch.from_numpy(np.stack(imgs)) 
    branch_ids = torch.tensor(branch_ids, dtype=torch.long)
    task_vec = torch.stack(task_vec)
    prevs = torch.stack(prevs)
    targets = torch.stack(targets)
    return imgs, branch_ids, task_vec, prevs, targets

