import numpy as np
import torch
from ProstateDataset import ProstateDataset 
from LITSDataset import LITSDataset 
from torch.utils.data import Dataset, DataLoader
from UNET import UNet
from TransformerNet import TransUNet
from PIL import Image 

def IOU(y_target, y_predict, reduce=False):
    '''
    y_target = H * W
    y_predict = H * W * C
    '''
    length = y_predict.size(1)
    if not reduce:
        y_arg = torch.argmax(y_predict, dim=1)
    else:
        y_arg = y_predict


    intersection = (torch.eq(y_arg, y_target) & (y_arg != 0)).sum(dim=[1, 2])
    union = torch.zeros_like(intersection)
    for l in range(1,length):
        union = union + (y_arg == l).sum(dim=[1,2])
        union = union + (y_target == l).sum(dim=[1,2])
    union = union - intersection
    
    iou = intersection.float() / torch.max(union.float(),torch.ones((16,1)))
    return iou.mean().item()

def precision(y_target, y_predict, reduce=False):
    '''
    y_target = H * W
    y_predict = H * W * C
    '''
    length = y_predict.size(1)
    if not reduce:
        y_arg = torch.argmax(y_predict, dim=1)
    else:
        y_arg = y_predict
    true_positives = ((y_arg == y_target) & (y_target != 0)).sum(dim=[1, 2])
    false_positives = ((y_arg != y_target) & (y_target == 0)).sum(dim=[1, 2])
    precision = true_positives.float() / torch.max((true_positives + false_positives).float(),torch.ones((16,1)))  

    return precision.mean().item()

def recall(y_target, y_predict, reduce=False):
    '''
    y_target = H * W
    y_predict = H * W * C
    '''
    length = y_predict.size(1)
    if not reduce:
        y_arg = torch.argmax(y_predict, dim=1)
    else:
        y_arg = y_predict
    true_positives = ((y_arg == y_target) & (y_target != 0)).sum(dim=[1, 2])
    false_negatives = ((y_arg != y_target) & (y_target != 0)).sum(dim=[1, 2])
    recall = true_positives.float() / torch.max((true_positives + false_negatives).float(),torch.ones((16,1)))

    return recall.mean().item()


def DICE(y_target, y_predict, reduce=False):
    '''
    y_target = H * Ws
    y_predict = H * W * C
    '''
    length = y_predict.size(1)
    if not reduce:
        y_arg = torch.argmax(y_predict, dim=1)
    else:
        y_arg = y_predict
    intersection = (torch.eq(y_arg, y_target) & (y_arg != 0)).sum(dim=[1, 2])
    union = torch.zeros_like(intersection)
    for l in range(1,length):
        union = union + (y_arg == l).sum(dim=[1,2])
        union = union + (y_target == l).sum(dim=[1,2])
    dice = 2 * intersection.float() / torch.max(union.float(),torch.ones((16,1)))
    return dice.mean().item()

def evaluate(dataset, path, network="UNET"):
    if dataset == "prostate":
        val_dataset = ProstateDataset(val_fold=0, validation=True, size=(256,256))
        if network == "UNET":
            size = (256,256)
            model = UNet(in_channels=3, out_channels=2)
        else:
            size = (128, 128)
            model = TransUNet(img_dim=128,
                            in_channels=3,
                            out_channels=128,
                            head_num=4,
                            mlp_dim=512,
                            block_num=8,
                            patch_dim=16,
                            class_num=2)
        val_dataset = ProstateDataset(val_fold=0, validation=True, size=size)
    else:
        if network == "UNET":
            size = (256, 256)
            model = UNet(in_channels=3, out_channels=3)
        else:
            size = (128, 128)
            model = TransUNet(img_dim=128,
                            in_channels=3,
                            out_channels=128,
                            head_num=4,
                            mlp_dim=512,
                            block_num=8,
                            patch_dim=16,
                            class_num=3)
        val_dataset = LITSDataset(val_fold=0, validation=True, size=size)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    model.load_state_dict(torch.load(path))
   
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
            iou_sum = 0
            dice_sum = 0
            recall_sum = 0
            precision_sum = 0
            iou_count = 0
            for data, targets in val_loader:
                inputs = data.to(device)
                

                outputs = model(inputs).cpu()

                iou_sum += IOU(y_target=targets, y_predict=outputs)
                dice_sum += DICE(y_target=targets, y_predict=outputs)
                recall_sum += recall(y_target=targets, y_predict=outputs)
                precision_sum += precision(y_target=targets, y_predict=outputs)
                iou_count += 1
                print(iou_sum, dice_sum)

    print("DICE:",dice_sum/iou_count)
    print("IOU:",iou_sum/iou_count)
    print("RECALL:",recall_sum/iou_count)
    print("Precision:",precision_sum/iou_count)


if __name__ == "__main__":
    #target = torch.randint(low=0, high=3, size=(16, 256, 256))
    #predicted = torch.randn((16,3,256,256))
    #print(IOU(target, predicted))
    #evaluate("lits","best_UNET_lits.pt", network="UNET")
    dataset = "prostate"
    network = "UNET"
    path = ""
    if dataset == "prostate":
        if network == "UNET":
            size = (256,256)
            model = UNet(in_channels=3, out_channels=2)
        else:
            size = (128, 128)
            model = TransUNet(img_dim=128,
                            in_channels=3,
                            out_channels=128,
                            head_num=4,
                            mlp_dim=512,
                            block_num=8,
                            patch_dim=16,
                            class_num=2)
    else:
        if network == "UNET":
            size = (256, 256)
            model = UNet(in_channels=3, out_channels=3)
        else:
            size = (128, 128)
            model = TransUNet(img_dim=128,
                            in_channels=3,
                            out_channels=128,
                            head_num=4,
                            mlp_dim=512,
                            block_num=8,
                            patch_dim=16,
                            class_num=3)
    #model.load_state_dict(torch.load(path))
   
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        image_path = "ProstateX/Processed/p0056slice7.png"

        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize(size)

        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()

        inputs = image.to(device)
                

        outputs = model(inputs).cpu()
        outputs = torch.argmax(outputs, dim=1) * 255
        outputs = torch.squeeze(outputs).numpy()

        # convert the tensor to a grayscale PIL image
        pil_image = Image.fromarray(outputs.astype('uint8'), mode='L')

        # save the image
        pil_image.save('output.jpg')
        print(outputs)