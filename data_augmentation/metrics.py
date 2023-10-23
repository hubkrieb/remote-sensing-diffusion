import os
import clip
from PIL import Image
import torch.nn as nn

def clip_image_similarity(image1_path, mask1_path, image2_path, mask2_path, model, preprocess, device):

    cos = nn.CosineSimilarity(dim = 0)

    image1 = Image.open(image1_path)
    mask1 = Image.open(mask1_path)
    min_x1, min_y1, max_x1, max_y1 = get_mask_coordinate(mask1)
    masked_image1 = Image.composite(image1, Image.new('RGB', (256, 256), (0, 0, 0)), mask1)
    masked_image1 = masked_image1.crop((min_x1, min_y1, max_x1, max_y1))
    image1_preprocess = preprocess(masked_image1).unsqueeze(0).to(device)
    image1_features = model.encode_image(image1_preprocess)

    image2 = Image.open(image2_path)
    mask2 = Image.open(mask2_path)
    min_x2, min_y2, max_x2, max_y2 = get_mask_coordinate(mask2)
    masked_image2 = Image.composite(image2, Image.new('RGB', (256, 256), (0, 0, 0)), mask2)
    masked_image2 = masked_image2.crop((min_x2, min_y2, max_x2, max_y2))
    image2_preprocess = preprocess(masked_image2).unsqueeze(0).to(device)
    image2_features = model.encode_image(image2_preprocess)

    similarity = cos(image1_features[0],image2_features[0]).item()

    return similarity

def average_similarity(images_path, images_masks_path, references_path, references_masks_path, device, verbose = False):
    
    model, preprocess = clip.load('ViT-B/32', device=device)

    avg_score = 0
    for image in os.listdir(images_path):
        img_avg_score = 0
        mask = image.split('-')[1]
        for reference in os.listdir(references_path):
            reference_mask = reference[:-10] + 'v1_' + reference[-10:]
            score = clip_image_similarity(os.path.join(images_path, image), os.path.join(images_masks_path, mask), os.path.join(references_path, reference), os.path.join(references_masks_path, reference_mask), model, preprocess, device)
            img_avg_score += score
        img_avg_score /= len(os.listdir(references_path))
        if verbose:
            print(f'Average score for image {image} :', img_avg_score)
        avg_score += img_avg_score
    avg_score /= len(os.listdir(images_path))

    return avg_score

def get_mask_coordinate(mask):
    width, height = mask.size
    min_x, min_y, max_x, max_y = width, height, -1, -1

    for x in range(width):
        for y in range(height):
            if mask.getpixel((x, y)) > 0:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    
    return min_x, min_y, max_x + 1, max_y + 1