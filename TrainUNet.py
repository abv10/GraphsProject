from UNET import UNet
from ProstateDataset import ProstateDataset 
from LITSDataset import LITSDataset 

from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(dataset="prostate", output_channels=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=output_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if dataset == "prostate":
        train_dataset = ProstateDataset(val_fold=0)
    else:
        train_dataset = LITSDataset(val_fold=0)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = ProstateDataset(val_fold=0, validation=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    num_epochs = 200
    best_val_loss = float('inf')
    early_stop = False
    patience = 3
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            inputs = data.to(device)
            targets = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                inputs = data.to(device)
                targets = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, targets.squeeze(dim=1))
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f'Epoch {epoch}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"best_{dataset}.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                early_stop = True
                break

    if early_stop:
        print("Training stopped early after ", epoch)
    else:
        print("Training completed")

if __name__ == "__main__":
    train(dataset="lits", output_channels=3)
