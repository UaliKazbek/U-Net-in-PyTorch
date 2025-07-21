import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, padding=1)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        out = self.act(x)
        return out

class DownSample(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv = DoubleConv(inp, out)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        out = self.pooling(x)
        return out, x

class UpSample(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.convt = nn.ConvTranspose2d(inp, out, 2, 2)
        self.conv = DoubleConv(inp, out)

    def forward(self, x1, x2):
        x = self.convt(x1)
        x = torch.cat([x, x2], dim=1)
        out = self.conv(x)
        return out


class UNet(nn.Module):
    def __init__(self, inp=3, num_classes=1):
        super().__init__()
        self.down1 = DownSample(inp, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.conv1 = DoubleConv(512, 1024)

        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        self.conv2 = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)
        x, x4 = self.down4(x)

        x = self.conv1(x)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.conv2(x)
        return out
