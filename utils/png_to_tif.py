import os
import argparse
import cv2
import rasterio
import numpy as np

def convert(input_path, output_path, profile_path):
    for input_name in os.listdir(input_path):
        input = cv2.imread(os.path.join(input_path, input_name))
        b, g, r = cv2.split(input)
        channel_image = np.zeros((input.shape[0], input.shape[1], 10), dtype=np.int32)
        channel_image[..., 6] = r
        channel_image[..., 5] = g
        channel_image[..., 1] = b
        channel_image = channel_image.transpose((2, 0, 1))
        background_name = input_name.split('-')[0] + '.tif'
        profile = rasterio.open(os.path.join(profile_path, background_name)).profile
        new_dataset = rasterio.open(
        os.path.join(output_path, input_name.split('.')[0] + '.tif'),
        'w',
        **profile
        )
        new_dataset.write(channel_image*257)
        new_dataset.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required = True, help = 'Path to the directory containing the PNG images to convert')
    parser.add_argument('--output_path', required = True, help = 'Path to the directory the TIF outputs will be stored')
    parser.add_argument('--profile_path', required = True, help = 'Path to the directory containing the original background profile (the original dataset landsat_patches directory)')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    profile_path = args.profile_path

    convert(input_path, output_path, profile_path)