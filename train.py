import metrics
import torch
import utils
import wandb
from torchmetrics.classification import BinaryJaccardIndex

def train(model, epochs, train_dataloader, val_dataloader, criterion, optimizer, checkpoint_interval, checkpoint_dir, device):
    
    model = model.to(device)
    
    wandb.run.define_metric(name = 'Training Loss', step_metric = 'epoch')
    wandb.run.define_metric(name = 'Validation Loss', step_metric = 'epoch')
    wandb.run.define_metric(name = 'IoU', step_metric = 'epoch')
    wandb.run.define_metric(name = 'Dice Coefficient', step_metric = 'epoch')
    wandb.run.define_metric(name = 'Pixel Accuracy', step_metric = 'epoch')
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
        
        if (epoch + 1) % checkpoint_interval == 0:
            val_loss, iou, dice, acc = evaluate(model, val_dataloader, criterion, device)
            wandb.log({'Training Loss' : train_loss, 'Validation Loss' : val_loss, 'IoU' : iou, 'Dice Coefficient' : dice, 'Pixel Accuracy' : acc, 'epoch' : epoch + 1})
            utils.save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)
            print('Epoch : {} | Train. Loss : {:.3f} | Val. Loss : {:.3f} | IoU : {:.3f} | Dice Coef. : {:.3f} | Pixel Acc. {:.3f}'.format(epoch + 1, train_loss, val_loss, iou, dice, acc))
        else:
            wandb.log({'Training Loss' : train_loss, 'epoch' : epoch + 1})

def evaluate(model, dataloader, criterion, device):
    
    model = model.to(device)
    
    model.eval()

    iou_metric = BinaryJaccardIndex().to(device)

    val_loss = 0
    iou = 0
    dice = 0
    acc = 0

    with torch.no_grad():
        for image, mask in dataloader:
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            loss = criterion(output, mask)
            pred = output > 0.5
            val_loss += loss.item()
            iou += iou_metric(pred, mask).item()
            dice += metrics.dice_coeff(pred, mask).item()
            acc += metrics.pixel_accuracy(pred, mask).item()
        
        val_loss /= len(dataloader)
        iou /= len(dataloader)
        dice /= len(dataloader)
        acc /= len(dataloader)

    return val_loss, iou, dice, acc


