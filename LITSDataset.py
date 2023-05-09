import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class LITSDataset(Dataset):
    def __init__(self, val_fold, validation=False, transform=None, size=(384,384)):
        self.size = size
        self.image_paths = []
        self.mask_paths = []
        if not validation:
            for i in range(5):
                if i != val_fold:
                    with open("lits_fold_"+str(i) + ".txt","r") as f:
                        for line in f:
                            self.mask_paths.append("LITS/Masks/"+str(line.strip()))
                            self.image_paths.append("LITS/Processed/"+str(line.strip()))
        else:
            i = val_fold
            with open("lits_fold_"+str(val_fold) + ".txt","r") as f:
                for line in f:
                    self.mask_paths.append("LITS/Masks/"+str(line.strip()))
                    self.image_paths.append("LITS/Processed/"+str(line.strip()))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        image = image.convert("RGB")
        new_size = self.size
        image = image.resize(new_size)

        mask = mask.resize(new_size, resample=Image.NEAREST)    
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        mask = np.array(mask)
        bins = np.array([0,127,155])
        mask = np.digitize(mask, bins) -1
        mask = torch.from_numpy(mask).long()

        image = torch.from_numpy(image).float()

        return image, mask

