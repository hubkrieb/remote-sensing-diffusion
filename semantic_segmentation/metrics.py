import torch

def dice_coeff(pred, target):
    intersection = torch.logical_and(pred, target)
    dice_score = 2 * torch.sum(intersection) / (torch.sum(pred) + torch.sum(target))
    return dice_score

def pixel_accuracy (target, pred):
    intersection = torch.sum(pred == target)
    total = torch.numel(target)
    if total == 0:
        pixel_accuracy = 0
    else:
        pixel_accuracy = intersection / total
    pixel_accuracy = intersection / total
    return pixel_accuracy  