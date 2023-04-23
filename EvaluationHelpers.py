import numpy as np
import torch
from ProstateDataset import ProstateDataset 
from LITSDataset import LITSDataset 
from torch.utils.data import Dataset, DataLoader
from UNET import UNet

def IOU(y_target, y_predict):
    '''
    y_target = H * W
    y_predict = H * W * C
    '''
    length = y_predict.size(1)
    y_arg = torch.argmax(y_predict, dim=1)
    intersection = (torch.eq(y_arg, y_target) & (y_arg != 0)).sum(dim=[1, 2])
    union = torch.zeros_like(intersection)
    for l in range(1,length):
        union = union + (y_arg == l).sum(dim=[1,2])
        union = union + (y_target == l).sum(dim=[1,2])
    union = union - intersection
    
    iou = intersection.float() / torch.max(union.float(),torch.ones((16,1)))
    return iou.mean().item()

def precision(y_target, y_predict):
    '''
    y_target = H * W
    y_predict = H * W * C
    '''
    length = y_predict.size(1)
    y_arg = torch.argmax(y_predict, dim=1)
    true_positives = ((y_arg == y_target) & (y_target != 0)).sum(dim=[1, 2])
    false_positives = ((y_arg != y_target) & (y_target != 0)).sum(dim=[1, 2])
    precision = true_positives.float() / (true_positives + false_positives).float()
    return precision.mean().item()

def recall(y_target, y_predict):
    '''
    y_target = H * W
    y_predict = H * W * C
    '''
    length = y_predict.size(1)
    y_arg = torch.argmax(y_predict, dim=1)
    true_positives = ((y_arg == y_target) & (y_target != 0)).sum(dim=[1, 2])
    false_negatives = ((y_arg != y_target) & (y_arg == 0)).sum(dim=[1, 2])
    recall = true_positives.float() / (true_positives + false_negatives).float()
    return recall.mean().item()


def DICE(y_target, y_predict):
    '''
    y_target = H * Ws
    y_predict = H * W * C
    '''
    length = y_predict.size(1)
    y_arg = torch.argmax(y_predict, dim=1)
    intersection = (torch.eq(y_arg, y_target) & (y_arg != 0)).sum(dim=[1, 2])
    union = torch.zeros_like(intersection)
    for l in range(1,length):
        union = union + (y_arg == l).sum(dim=[1,2])
        union = union + (y_target == l).sum(dim=[1,2])
    dice = 2 * intersection.float() / union.float()
    return dice.mean().item()

def evaluate(dataset, path):
    if dataset == "prostate":
        val_dataset = ProstateDataset(val_fold=0, validation=True)
        model = UNet(in_channels=3, out_channels=2)

    else:
        val_dataset = LITSDataset(val_fold=0, validation=True)
        model = UNet(in_channels=3, out_channels=3)


    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    model.load_state_dict(torch.load(path))
   
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
            iou_sum = 0
            dice_sum = 0
            iou_count = 0
            for data, targets in val_loader:
                inputs = data.to(device)
                

                outputs = model(inputs).cpu()

                iou_sum += IOU(y_target=targets, y_predict=outputs)
                dice_sum += DICE(y_target=targets, y_predict=outputs)
                iou_count += 1
                print(iou_sum, iou_count)

    print("DICE:",dice_sum/iou_count)
    print("IOU:",iou_sum/iou_count)



if __name__ == "__main__":
    #target = torch.randint(low=0, high=3, size=(16, 256, 256))
    #predicted = torch.randn((16,3,256,256))
    #print(IOU(target, predicted))
    evaluate("prostate","best_prostate.pt")



        
