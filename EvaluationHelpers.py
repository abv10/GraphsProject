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
    y_arg = torch.argmax(y_predict, dim=2)
    intersection = 0
    union = 0

    for c in range(1,y_predict.size[2]): #Don't need the background
        for row in y_arg:
            for col in y_arg[row]:
                yp = y_arg[row][col]
                yt = y_target[row][col]
                if yp == c and yt == c:
                    intersection += 1
                elif yp == c or yt == c:
                    union += 1

    return intersection / union

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

    with torch.no_grad():
            iou_sum = 0
            iou_count = 0
            for data, labels in val_loader:
                inputs = data.to(device)
                targets = labels.to(device)

                outputs = model(inputs)

                for d in range(inputs.size[1]):
                    iou_sum += IOU(y_target=targets[d], y_predict=outputs[d])
                    iou_count += 1

    print(iou_sum/iou_count)


if __name__ == "__main__":
    evaluate("prostate","best_prostate.pt")



        
