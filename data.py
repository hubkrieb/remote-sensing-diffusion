import os
import torch
from torch.utils import data
import utils

class SegmentationDataset(data.Dataset):
        def __init__(self, image_folder, mask_folder, n_channels = 3):
            self.image_folder = image_folder
            self.mask_folder = mask_folder

            self.n_channels = n_channels

            self.images = os.listdir(image_folder)
            self.masks = os.listdir(mask_folder)
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image_path = os.path.join(self.image_folder, self.images[idx])
            mask_path = os.path.join(self.mask_folder, self.masks[idx])

            image = utils.get_img_arr(image_path, self.n_channels)
            mask = utils.get_mask_arr(mask_path)

            return image, mask

def get_dataloaders(image_folder, mask_folder, n_channels, batch_size, split = [0.7, 0.15, 0.15], num_workers = 0):
    """ Generates Training, Validation and Test Dataloaders"""
    generator = torch.Generator().manual_seed(42)
    dataset = SegmentationDataset(image_folder, mask_folder, n_channels)
    train_dataset, val_dataset, test_dataset = data.random_split(dataset, split, generator)
    train_dataloader = data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_dataloader = data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    test_dataloader = data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    return train_dataloader, val_dataloader, test_dataloader