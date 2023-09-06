import data
import model
import train
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs')
    parser.add_argument('--batch_size')
    parser.add_argument('--n_channels')
    parser.add_argument('--checkpoint_interval')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--split_path')
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    n_channels = args.n_channels
    checkpoint_interval = args.checkpoint_interval
    checkpoint_path = args.checkpoint_path
    split_path = args.split_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = data.get_dataloader('dataset/augmented_dataset/lmdb', n_channels, batch_size, split_path, 'train', True)
    val_dataloader = data.get_dataloader('dataset/augmented_dataset/lmdb', n_channels, batch_size, split_path, 'val', False)
    test_dataloader = data.get_dataloader('dataset/augmented_dataset/lmdb', n_channels, batch_size, split_path, 'test', False)

    unet = model.Unet(n_channels)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(unet.parameters())

    wandb.init(project = 'Diffusion Model based Data Augmentation for Remote Sensing Imagery')

    train.train(unet, epochs, train_dataloader, val_dataloader, criterion, optimizer, checkpoint_interval, checkpoint_path, device)