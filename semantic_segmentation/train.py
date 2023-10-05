import os
import sys
import metrics
import torch
import wandb
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall

current = os.path.dirname(os.path.realpath('__file__'))
sys.path.append(current)

import utils

def train(model, epochs, train_dataloader, val_dataloader, criterion, optimizer, checkpoint_interval, checkpoint_dir, device):
    
    model = model.to(device)
    
    wandb.run.define_metric(name = 'Training Loss', step_metric = 'epoch')
    wandb.run.define_metric(name = 'Validation Loss', step_metric = 'epoch')
    wandb.run.define_metric(name = 'IoU', step_metric = 'epoch')
    wandb.run.define_metric(name = 'Dice Coefficient', step_metric = 'epoch')
    wandb.run.define_metric(name = 'Pixel Accuracy', step_metric = 'epoch')
    wandb.run.define_metric(name = 'Precision', step_metric = 'epoch')
    wandb.run.define_metric(name = 'Recall', step_metric = 'epoch')
    wandb.run.define_metric(name = 'epoch', hidden = True)

    for epoch in range(epochs):
        model.train()

        train_loss = 0

        for image, mask in train_dataloader:
            optimizer.zero_grad()
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        train_loss /= len(train_dataloader)
        
        if epoch % checkpoint_interval == 0:
            val_loss, iou, dice, acc, prec, rec = evaluate(model, val_dataloader, criterion, device)
            wandb.log({'Training Loss' : train_loss, 'Validation Loss' : val_loss, 'IoU' : iou, 'Dice Coefficient' : dice, 'Pixel Accuracy' : acc, 'Precision' : prec, 'Recall' : rec, 'epoch' : epoch + 1})
            utils.save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)
            print('Epoch : {} | Train. Loss : {:.4f} | Val. Loss : {:.4f} | IoU : {:.4f} | Dice Coef. : {:.4f} | Pixel Acc. {:.4f} | Precision {:.4f} | Recall {:.4f}'.format(epoch + 1, train_loss, val_loss, iou, dice, acc, prec, rec))
        else:
            wandb.log({'Training Loss' : train_loss, 'epoch' : epoch + 1})

def evaluate(model, dataloader, criterion, device):
    
    model = model.to(device)
    
    model.eval()

    iou_metric = BinaryJaccardIndex().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)

    val_loss = 0
    iou = 0
    dice = 0
    acc = 0
    prec = 0
    rec = 0

    non_zero_fp = 0

    with torch.no_grad():
        for image, mask in dataloader:
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            loss = criterion(output, mask)
            pred = output > 0.5
            val_loss += loss.item()
            acc += metrics.pixel_accuracy(pred, mask).item()

            # IoU and Dice score are not defined when the target is zeros
            if mask.sum() > 0:
                iou += iou_metric(pred, mask).item()
                dice += metrics.dice_coeff(pred, mask).item()
                prec += precision_metric(pred, mask).item()
                rec += recall_metric(pred, mask).item()
                non_zero_fp += 1
        
        val_loss /= len(dataloader)
        acc /= len(dataloader)
        prec /= non_zero_fp
        rec /= non_zero_fp
        iou /= non_zero_fp
        dice /= non_zero_fp

    return val_loss, iou, dice, acc, prec, rec


