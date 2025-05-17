import os
import h5py
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


# OS enviroment

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- Data loading ---
class GlitchDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:]
        data = np.array([
            (img - np.min(img)) / (np.max(img) - np.min(img)) if np.max(img) > np.min(img) else np.zeros_like(img)
            for img in data
        ])
        data = np.expand_dims(data, axis=1).astype(np.float32)  # shape: (N, 1, H, W)
        data += 1e-6
        self.data = torch.tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # Autoencoder target = input

# --- Autoencoder Model ---
class ConvAutoencoder(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.1),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2),
            nn.Dropout(0.1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.Dropout(0.1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        val_loss = F.mse_loss(x_hat, y)        
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_end(self):
        
        best_model = self.trainer.checkpoint_callback.best_model_path
        torch.save(best_model, "best_model.ckpt")


# --- Training script ---
if __name__ == "__main__":
    data_path = Path("/data/yiyang/ML/unsupervised/glitches/K1/glitchs_HFP4.h5")
    dataset = GlitchDataset(data_path)

    val_split = int(0.2 * len(dataset))
    train_data, val_data = random_split(dataset, [len(dataset) - val_split, val_split])
    train_loader = DataLoader(
        train_data, 
        batch_size=512, 
        shuffle=True
    )
    val_loader = DataLoader(val_data, batch_size=32)

    model = ConvAutoencoder()

    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=[
            EarlyStopping(
                monitor='val_loss', 
                patience=3, 
                mode='min')
            ],
        accelerator='gpu',
        strategy="auto"   
    )

    trainer.fit(model, train_loader, val_loader)