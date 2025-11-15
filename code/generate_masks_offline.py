import os
import shutil
import torch
import numpy as np
import argparse
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.slic import generate_sgements_slic_based
from scipy.ndimage import rotate, zoom
import random
import h5py
from torch.utils.data import Dataset


parser = argparse.ArgumentParser()
# /home/chengboding/data/ACDC  /home/chengboding/data/synapse
parser.add_argument('--root_path', type=str, default='/data/chengboding/data/synapse')
parser.add_argument('--dataset', type=str, default='Synapse', help='dataset name')
parser.add_argument('--n_segments', type=int,  default=60, help='n_segments')   # ACDC:40   Synapase:60
parser.add_argument('--compactness', type=float, default=0.1, help='compactness')  # ACDC:0.1   Synapase:0.1
args = parser.parse_args()

save_dir = f'slic_mask_cache_{args.dataset}'

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, dataset=None):
        self._base_dir = base_dir
        self.dataset = dataset
        self.sample_list = []
        if dataset == "ACDC":
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
        elif dataset == "Synapse":
            with open(self._base_dir + '/lists/lists_Synapse/train.txt', 'r') as f1:
                self.sample_list = f1.readlines()
        else:
            raise NotImplementedError("Dataset Not Implemented")

        self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.dataset == "ACDC":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        elif self.dataset == "Synapse":
            h5f = h5py.File(self._base_dir + "/h5_2d/{}.h5".format(case), 'r')
        else:
            raise NotImplementedError("Dataset Not Implemented")
 
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        # sample["idx"] = idx
        sample['case'] = case
        return sample
    

def generate_all_masks():
    db_train = BaseDataSets(base_dir=args.root_path, dataset=args.dataset)
    dataloader = DataLoader(db_train, batch_size=1, shuffle=False, num_workers=4)

    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        image = data['image'].unsqueeze(0)
        case = data['case']
        print(case)

        segments = generate_sgements_slic_based(image, n_segments=args.n_segments, compactness=args.compactness)
        segments = segments.squeeze(0)
        print(segments.shape)

        unique_values = torch.unique(segments)

        print("Segments shape:", segments.shape)
        print("Unique values:", unique_values)
        print("Number of unique values:", len(unique_values))
        torch.save(segments.cpu(), os.path.join(save_dir, f"segments_{case[0]}.pt"))

if __name__ == "__main__":
    generate_all_masks()
