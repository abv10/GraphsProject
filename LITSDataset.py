import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class LITSDataset(Dataset):
    def __init__(self, val_fold, validation=False, transform=None):
        self.image_paths = []
        self.mask_paths = []
        if not validation:
            for i in range(5):
                if i != val_fold:
                    with open("lits_fold_"+str(i) + ".txt","r") as f:
                        for line in f:
                            self.mask_paths.append("LITS\\Masks\\"+str(line.strip()))
                            self.image_paths.append("LITS\\Processed\\"+str(line.strip()))
        else:
            i = val_fold
            with open("lits_fold_"+str(val_fold) + ".txt","r") as f:
                for line in f:
                    self.mask_paths.append("LITS\\Masks\\"+str(line.strip()))
                    self.image_paths.append("LITS\\Processed\\"+str(line.strip()))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        image = image.convert("RGB")
        new_size = (256,256)
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




#train_dataset = CustomDataset(train_file_paths, transform=transform)
#train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#val_dataset = CustomDataset(val_file_paths, transform=transform)
#val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


if __name__ == "__main__":
    # Create an instance of the dataset
    dataset = ProstateDataset(val_fold=0)

    # Call __getitem__ with index 0
    for i in range(3):
        image, mask = dataset.__getitem__(i)

    # Print the results
        print(image.shape)  # should be (height, width, channels)
        print(np.max(mask))   # should be (height, width)