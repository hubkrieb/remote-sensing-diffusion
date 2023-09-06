import os
import sys
import argparse
import shutil
import pandas as pd

current = os.path.dirname(os.path.realpath('__file__'))
sys.path.append(current)

from utils import get_mask_arr

def augment(input_path, output_path, mask_path, output_mask_path, split_path, output_split_path):   
    train = pd.read_csv(os.path.join(split_path, 'train.csv'), index_col = None, header = None)
    for input_name in os.listdir(input_path):
        mask_name = input_name.split('-')[1]
        new_mask_name = input_name[:-10] + 'v1_' + input_name[-10:]
        mask = get_mask_arr(os.path.join(mask_path, mask_name))
        train = train.append(pd.Series([input_name.split('.')[0], int(mask.sum())]), ignore_index = True)
        shutil.copy2(os.path.join(mask_path, mask_name), os.path.join(output_mask_path, new_mask_name))
        shutil.copy2(os.path.join(input_path, input_name), os.path.join(output_path, input_name))
    shutil.copy(os.path.join(split_path, 'test.csv'), os.path.join(output_split_path, 'test.csv'))
    shutil.copy(os.path.join(split_path, 'val.csv'), os.path.join(output_split_path, 'val.csv'))
    train.to_csv(os.path.join(output_split_path, 'train.csv'), index = None, header = None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument('--mask_path')
    parser.add_argument('--output_mask_path')
    parser.add_argument('--split_path')
    parser.add_argument('--output_split_path')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    mask_path = args.mask_path
    output_mask_path = args.output_mask_path
    split_path = args.split_path
    output_split_path = args.output_split_path

    augment(input_path, output_path, mask_path, output_mask_path, split_path, output_split_path)
