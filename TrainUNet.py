from UNET import UNet
from ProstateDataset import ProstateDataset 
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_dataset = ProstateDataset(val_fold=0)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = ProstateDataset(val_fold=0, validation=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
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
        for batch_idx, data in enumerate(val_loader):
            inputs = data.to(device)
            targets = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f'Epoch {epoch}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')
