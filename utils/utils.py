import rasterio
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

MAX_PIXEL_VALUE = 65535

def get_img_arr(path, n_channels):
    if n_channels == 3:
        img = rasterio.open(path).read((7,6,2))
    elif n_channels == 10:
        img = rasterio.open(path).read()
    img = np.float32(img) / MAX_PIXEL_VALUE 
    return torch.from_numpy(img)

def get_mask_arr(path):
    img = rasterio.open(path).read()
    seg = np.float32(img)
    return torch.from_numpy(seg)

def plot_prediction(model, image, mask, device):

    batch_size = image.shape[0]

    model = model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = output > 0.5

        pred = pred.cpu()
        image = image.cpu()
        mask = mask.cpu()

    fig, ax = plt.subplots(batch_size, 3, figsize = (20, 60))
    for i in range(batch_size):
        ax[i, 0].set_title('Image')
        ax[i, 1].set_title('Mask')
        ax[i, 2].set_title('Prediction')

        ax[i, 0].imshow(image[i].permute(1, 2, 0))
        ax[i, 1].imshow(mask[i].permute(1, 2, 0))
        ax[i, 2].imshow(pred[i].permute(1, 2, 0))

    plt.axis('off')
    plt.show()