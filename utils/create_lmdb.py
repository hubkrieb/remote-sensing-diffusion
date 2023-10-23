import lmdb
import sys
import os
import numpy as np
import argparse
import utils

def write_lmdb(image_folder, mask_folder, lmdb_path):
    # Set up LMDB environment
    env = lmdb.open(lmdb_path, map_size = int(2e10))
    txn = env.begin(write=True)
    
    image_names = os.listdir(image_folder)
    mask_names = os.listdir(mask_folder)

    # Iterate over image files in folder
    for image_name in image_names:
        mask_name = image_name[:-10] + 'v1_' + image_name[-10:]
        image_path = os.path.join(image_folder, image_name)
        assert os.path.exists(image_path), f'Image at {image_path} does not exist'
        image_key = image_name.split('.')[0]
        mask_path = os.path.join(mask_folder, mask_name)
        mask_key = image_key + '_mask'
        image = utils.get_img_arr(image_path, 3)
        # If the mask does not exist, then create an empty one
        if mask_name in mask_names:
            assert os.path.exists(mask_path), f'Mask at {mask_path} does not exist'
            mask = utils.get_mask_arr(mask_path)
            mask = mask.numpy().tobytes()
        else:
            mask = np.zeros((1, 256, 256), dtype = np.float32).tobytes()
        txn.put(image_key.encode('ascii'), image.numpy().tobytes())
        txn.put(mask_key.encode('ascii'), mask)
    
    # Commit changes and close LMDB environment
    txn.commit()
    env.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--background_path', required = True, help = 'Path to the directory containing the backgrounds')
    parser.add_argument('--mask_path', required = True, help = 'Path to the directory containing the masks')
    parser.add_argument('--output_path', required = True, help = 'Path to the directory to store the lmdb database')
    args = parser.parse_args()

    background_path = args.background_path
    mask_path = args.mask_path
    output_path = args.output_path

    write_lmdb(background_path, mask_path, output_path)