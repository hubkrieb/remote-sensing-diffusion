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

def average_similarity(image_path, references_path, model, preprocess, device):
    
    model, preprocess = clip.load('ViT-B/32', device=device)

    avg_score = 0

    for reference in os.listdir(references_path):
        score = clip_image_similarity(image_path, os.path.join(references_path, reference), model, preprocess, device)
        avg_score += score
    avg_score /= len(os.listdir(references_path))

    return avg_score