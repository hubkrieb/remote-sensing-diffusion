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

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):

    checkpoint = {'epoch' : epoch, 'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint{epoch}.pt')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch'] - 1
    return model, optimizer, epoch

def plot_prediction(model, image, mask, device):

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

    fig, ax = plt.subplots(1, 3, figsize = (10, 30))

    ax[0].set_title('Image')
    ax[1].set_title('Mask')
    ax[2].set_title('Prediction')

    ax[0].imshow(image[0].permute(1, 2, 0))
    ax[1].imshow(mask[0].permute(1, 2, 0))
    ax[2].imshow(pred[0].permute(1, 2, 0))

    plt.axis('off')
    plt.show()