"""
Pytorch Lightning DenseNet and AttentionDenseNet implementation.
"""

import torch
import lightning as L

from torch import nn, optim


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 9), padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 9), padding='same')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(2*out_channels, out_channels, kernel_size=(1, 9), padding='same')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(3*out_channels, out_channels, kernel_size=(1, 9), padding='same')
        self.relu4 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(out1))
        out3 = torch.cat([out1, out2], dim=1)
        out4 = self.relu3(self.conv3(out3))
        out5 = torch.cat([out3, out4], dim=1)
        out6 = self.relu4(self.conv4(out5))
        out = self.bn1(out6)
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, in_channels//2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_channels//2, in_channels//4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_channels//4, in_channels//8)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(in_channels//8, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


class DenseNet(L.LightningModule):
    def __init__(self, in_features=2):
        super().__init__()

        self.denseBlock1 = DenseBlock(in_features, 16)
        self.ap1 = nn.AvgPool2d(kernel_size=(1, 5))
        self.denseBlock2 = DenseBlock(16, 16)
        self.ap2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.denseBlock3 = DenseBlock(16, 4)
        self.denseBlock4 = DenseBlock(4, 4)
        self.ap3 = nn.AvgPool2d(kernel_size=(1, 5))
        self.flat1 = nn.Flatten()
        self.fc = LinearBlock(512, 2)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.denseBlock1(x)
        x = self.ap1(x)
        x = self.denseBlock2(x)
        x = self.ap2(x)
        x = self.denseBlock3(x)
        x = self.denseBlock4(x)
        x = self.ap3(x)
        x = self.flat1(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = self.euclidean_distance(y_hat, y)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, sync_dist=True)
        self.logger.log_metrics({'train_loss': loss}, step=self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = torch.cdist(y_hat, y).mean()
        self.log('val_loss', loss, sync_dist=True)
        self.logger.log_metrics({'val_loss': loss}, step=self.global_step)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.cdist(y_hat, y).mean()
        self.log('test_loss', loss, sync_dist=True)
        # When testing on the dynamic scenarios, change name of logged test loss here (to, e.g., test_loss_dynamic)
        self.logger.log_metrics({'test_loss': loss}, step=self.global_step)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, in_channels, bias=False)
        self.key = nn.Linear(in_channels, in_channels, bias=False)
        self.value = nn.Linear(in_channels, in_channels, bias=False)
        self.attention = nn.MultiheadAttention(in_channels, batch_first=True, num_heads=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        out, weights = self.attention(Q, K, V, need_weights=True)
        self.current_weights = weights
        out = out.transpose(1, 2)
        return out

    def get_attention_weights(self):
        return self.current_weights


class AttentionDenseNet(L.LightningModule):
    def __init__(self, in_features=2, num_antennas=64, num_subcarriers=100):
        super().__init__()

        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        
        self.denseBlock1 = DenseBlock(in_features, 16)
        self.subcarrier_attentions = nn.ModuleList([SelfAttention(16)
                                                   for _ in range(self.num_antennas)])
        self.ap1 = nn.AvgPool2d(kernel_size=(1, 5))
        self.denseBlock2 = DenseBlock(16, 16)
        self.ap2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.denseBlock3 = DenseBlock(16, 4)
        self.denseBlock4 = DenseBlock(4, 4)
        self.ap3 = nn.AvgPool2d(kernel_size=(1, 5))
        self.flat1 = nn.Flatten(end_dim=2)
        self.antenna_attention = SelfAttention(8)
        self.flat2 = nn.Flatten()
        self.fc = LinearBlock(512, 2)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.denseBlock1(x)  # [32, 16, 64, 100]
        x = [self.subcarrier_attentions[i](x[:, :, i, :]).unsqueeze(2) for i in range(self.num_antennas)]  # 64 x [32, 16, 1, 100]
        x = torch.cat(x, dim=2)  # [32, 16, 64, 100]
        x = self.ap1(x)  # [32, 16, 64, 20]
        x = self.denseBlock2(x)  # [32, 16, 64, 20]
        x = self.ap2(x)  # [32, 16, 64, 10]
        x = self.denseBlock3(x)  # [32, 4, 64, 10]
        x = self.denseBlock4(x)  # [32, 4, 64, 10]
        x = self.ap3(x)  # [32, 4, 64, 2]
        x = self.flat1(x.transpose(2,3))  # [32, 8, 64]
        x = self.antenna_attention(x)  # [32, 8, 64]
        x = self.flat2(x)  # [32, 512]
        x = self.fc(x)  # [32, 2]
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, sync_dist=True)
        self.logger.log_metrics({'train_loss': loss}, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = torch.cdist(y_hat, y).mean()
        self.log('val_loss', loss, sync_dist=True)
        self.logger.log_metrics({'val_loss': loss}, step=self.global_step)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.cdist(y_hat, y).mean()
        # When testing on the dynamic scenarios, change name of logged test loss here (to, e.g., test_loss_dynamic)
        self.log('test_loss', loss, sync_dist=True)
        self.logger.log_metrics({'test_loss': loss}, step=self.global_step)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
