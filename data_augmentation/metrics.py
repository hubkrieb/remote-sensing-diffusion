import os
import clip
from PIL import Image
import torch.nn as nn

def clip_image_similarity(image1_path, image2_path, model, preprocess, device):

    cos = nn.CosineSimilarity(dim = 0)

    image1_preprocess = preprocess(Image.open(image1_path)).unsqueeze(0).to(device)
    image1_features = model.encode_image( image1_preprocess)

    image2_preprocess = preprocess(Image.open(image2_path)).unsqueeze(0).to(device)
    image2_features = model.encode_image( image2_preprocess)

    similarity = cos(image1_features[0],image2_features[0]).item()

    return similarity

def average_similarity(images_path, references_path, device, verbose = False):
    
    model, preprocess = clip.load('ViT-B/32', device=device)

    avg_score = 0
    for image in os.listdir(images_path):
        img_avg_score = 0
        for reference in os.listdir(references_path):
            score = clip_image_similarity(os.path.join(images_path, image), os.path.join(references_path, reference), model, preprocess, device)
            img_avg_score += score
        img_avg_score /= len(os.listdir(references_path))
        if verbose:
            print(f'Average score for image {image} :', img_avg_score)
        avg_score += img_avg_score
    avg_score /= len(os.listdir(images_path))

    return avg_score