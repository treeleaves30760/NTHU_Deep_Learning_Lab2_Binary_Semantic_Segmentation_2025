import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResNet34_UNet(nn.Module):
    def __init__(self, n_classes, pretrained=True, bilinear=True):
        super(ResNet34_UNet, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 使用 ResNet34 作為編碼器
        resnet = models.resnet34(pretrained=pretrained)

        # 編碼器 (ResNet34 層)
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 通道
        self.pool = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64 通道
        self.encoder3 = resnet.layer2  # 128 通道
        self.encoder4 = resnet.layer3  # 256 通道
        self.encoder5 = resnet.layer4  # 512 通道

        # 解碼器 (UNet 風格的上採樣與跳躍連接)
        self.up1 = Up(512 + 256, 256, bilinear)
        self.up2 = Up(256 + 128, 128, bilinear)
        self.up3 = Up(128 + 64, 64, bilinear)
        self.up4 = Up(64 + 64, 64, bilinear)

        # 最終輸出層
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 保存原始輸入大小，用於最終上采樣
        input_size = x.size()[2:]

        # 編碼器
        e1 = self.encoder1(x)         # 64 通道, 1/2 解析度
        e1_pool = self.pool(e1)       # 64 通道, 1/4 解析度
        e2 = self.encoder2(e1_pool)   # 64 通道, 1/4 解析度
        e3 = self.encoder3(e2)        # 128 通道, 1/8 解析度
        e4 = self.encoder4(e3)        # 256 通道, 1/16 解析度
        e5 = self.encoder5(e4)        # 512 通道, 1/32 解析度

        # 解碼器與跳躍連接
        x = self.up1(e5, e4)  # 256 通道, 1/16 解析度
        x = self.up2(x, e3)   # 128 通道, 1/8 解析度
        x = self.up3(x, e2)   # 64 通道, 1/4 解析度
        x = self.up4(x, e1)   # 64 通道, 1/2 解析度

        # 獲取最終預測
        logits = self.outc(x)

        # 確保輸出和輸入圖像大小一致
        if logits.size()[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode='bilinear', align_corners=True)

        return logits
