import numpy as np
import torch
from ProstateDataset import ProstateDataset 
from LITSDataset import LITSDataset 
from torch.utils.data import Dataset, DataLoader
from UNET import UNet

def IOU_2(y_target, y_predict):
    '''
    y_target = H * W 
    y_predict = H * W * C
    '''
    y_arg = torch.argmax(y_predict, dim=0)
    intersection = 0
    union = 0

    for c in range(1,y_predict.size(2)): #Don't need the background
        for row in range(y_arg.size(0)):
            for col in range(y_arg.size(1)):
                yp = y_arg[row][col].item()
                yt = y_target[row][col].item()
                if yp == c and yt == c:
                    intersection += 1
                    union += 1
                elif yp == c or yt == c:
                    union += 1
    
    return intersection / union

def IOU(y_target, y_predict):
    '''
    y_target = H * W
    y_predict = H * W * C
    '''
    length = y_predict.size(1)
    y_arg = torch.argmax(y_predict, dim=1)
    print(y_arg.shape)
    print(y_target.shape)
    print(y_arg)
    print(y_target)
    intersection = (torch.eq(y_arg, y_target) & (y_arg != 0)).sum(dim=[1, 2])
    union = torch.zeros_like(intersection)
    for l in range(1,length):
        union = union + (y_arg == l).sum(dim=[1,2])
        union = union + (y_target == l).sum(dim=[1,2])
    union = union - intersection
    iou = intersection.float() / union.float()
    return iou.mean().item()

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
            iou_count = 0
            for data, targets in val_loader:
                inputs = data.to(device)
                

                outputs = model(inputs).cpu()

                iou_sum += IOU(y_target=targets, y_predict=outputs)
                iou_count += 1

    print(iou_sum/iou_count)


if __name__ == "__main__":
    #target = torch.randint(low=0, high=3, size=(16, 256, 256))
    #predicted = torch.randn((16,3,256,256))
    #print(IOU(target, predicted))
    evaluate("prostate","best_prostate.pt")



        
