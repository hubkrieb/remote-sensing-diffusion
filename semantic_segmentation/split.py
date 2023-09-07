import os
import sys
import numpy as np
import torch.utils.data as data
import torch
import argparse

current = os.path.dirname(os.path.realpath('__file__'))
sys.path.append(current)

from utils import get_img_arr, get_mask_arr

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

            image = get_img_arr(image_path, self.n_channels)

            mask_name = self.images[idx][:-10] + 'v1_' + self.images[idx][-10:]
            if mask_name in self.masks:
                mask_path = os.path.join(self.mask_folder, mask_name)
                mask = get_mask_arr(mask_path)
            else:
                mask = torch.zeros((1, 256, 256))

            return image, mask

def split_data(input_path, mask_path, output_path):

    inputs = np.asarray(os.listdir(input_path))
    inputs = np.char.rstrip(inputs, '.tif')

    dataset = SegmentationDataset(input_path, mask_path)
    dataloader = data.DataLoader(dataset)
    fire_pixels = []
    for _, mask in dataloader:
        fire_pixels.append(int(mask.sum()))
    inputs = np.array((inputs, fire_pixels)).T
    np.savetxt(os.path.join(output_path, 'fire_pixels.csv'), inputs, delimiter = ',', fmt = '%s')

    train_dataset, val_dataset, test_dataset = data.random_split(dataset, [0.7, 0.2, 0.1])

    train_inputs = inputs[train_dataset.indices]
    val_inputs = inputs[val_dataset.indices]
    test_inputs = inputs[test_dataset.indices]

    np.savetxt(os.path.join(output_path, 'train.csv'), train_inputs, delimiter = ',', fmt = '%s')
    np.savetxt(os.path.join(output_path, 'val.csv'), val_inputs, delimiter = ',', fmt = '%s')
    np.savetxt(os.path.join(output_path, 'test.csv'), test_inputs, delimiter = ',', fmt = '%s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--mask_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    input_path = args.input_path
    mask_path = args.mask_path
    output_path = args.output_path

    split_data(input_path, mask_path, output_path)