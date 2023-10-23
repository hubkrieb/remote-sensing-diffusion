import os
import sys
import argparse
import shutil
import pandas as pd

current = os.path.dirname(os.path.realpath('__file__'))
sys.path.append(current)

from utils.utils import get_mask_arr

def augment(input_path, output_path, mask_path, output_mask_path, original_image_path, original_mask_path, split_path, output_split_path):   
    print('Copying original dataset')
    #shutil.copytree(original_image_path, output_path)
    #shutil.copytree(original_mask_path, output_mask_path)
    print('Adding synthetic data')
    train = pd.read_csv(os.path.join(split_path, 'train.csv'), index_col = None, header = None)
    for input_name in os.listdir(input_path):
        mask_name = input_name.split('-')[1]
        new_mask_name = input_name[:-10] + 'v1_' + input_name[-10:]
        mask = get_mask_arr(os.path.join(mask_path, mask_name))
        train = train.append(pd.Series([input_name.split('.')[0], int(mask.sum())]), ignore_index = True)
        shutil.copy2(os.path.join(mask_path, mask_name), os.path.join(output_mask_path, new_mask_name))
        shutil.copy2(os.path.join(input_path, input_name), os.path.join(output_path, input_name))
    print('Creating splits')
    os.makedirs(output_split_path)
    shutil.copy(os.path.join(split_path, 'test.csv'), os.path.join(output_split_path, 'test.csv'))
    shutil.copy(os.path.join(split_path, 'val.csv'), os.path.join(output_split_path, 'val.csv'))
    train.to_csv(os.path.join(output_split_path, 'train.csv'), index = None, header = None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help = 'Path to synthetic images (tif format) to add to the original dataset')
    parser.add_argument('--output_path', help = 'Path to the augmented dataset images')
    parser.add_argument('--mask_path', help = 'Path to the masks used to generate synthetic data')
    parser.add_argument('--output_mask_path', help = 'Path to the augmented dataset masks')
    parser.add_argument('--original_image_path', help = 'Path to the original dataset images')
    parser.add_argument('--original_mask_path', help = 'Path to the original dataset masks')
    parser.add_argument('--split_path', help = 'Path to the original dataset split')
    parser.add_argument('--output_split_path', help = 'Path to the augmented dataset split')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    mask_path = args.mask_path
    output_mask_path = args.output_mask_path
    original_image_path = args.original_image_path
    original_mask_path = args.original_mask_path
    split_path = args.split_path
    output_split_path = args.output_split_path

    augment(input_path, output_path, mask_path, output_mask_path, original_image_path, original_mask_path, split_path, output_split_path)
