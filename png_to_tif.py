import os
import argparse
import cv2
import rasterio
import numpy as np

def convert(input_path, output_path, profile_path, dsize):
    for input_name in os.listdir(input_path):
        input = cv2.imread(os.path.join(input_path, input_name))
        if dsize:
            input = cv2.resize(input, (dsize, dsize))
        b, g, r = cv2.split(input)
        channel_image = np.zeros((input.shape[0], input.shape[1], 10), dtype=np.int32)
        channel_image[..., 6] = r
        channel_image[..., 5] = g
        channel_image[..., 1] = b
        channel_image = channel_image.transpose((2, 0, 1))
        background_name = input_name.split('-init')[0] + '.tif'
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
    parser.add_argument('--input_path', required = True)
    parser.add_argument('--output_path', required = True)
    parser.add_argument('--profile_path', required = True)
    parser.add_argument('--size', required = False)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    profile_path = args.profile_path
    size = args.size

    convert(input_path, output_path, profile_path, size)