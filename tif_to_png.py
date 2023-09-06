import os
import argparse
from torchvision.transforms.functional import to_pil_image
from utils import get_img_arr, get_mask_arr

def convert(input_path, output_path, image):
    inputs = os.listdir(input_path)
    for input_name in inputs:
        if image == 'mask':
            input = to_pil_image(get_mask_arr(os.path.join(input_path, input_name)))
        elif image == 'background':
            input = to_pil_image(get_img_arr(os.path.join(input_path, input_name), 3))
        input.save(os.path.join(output_path, input_name.split('.')[0] + '.png'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument('--image')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    image = args.image

    convert(input_path, output_path, image)