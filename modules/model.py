import torch
from torch import nn
import segmentation_models_pytorch as smp


class CustomCNN(nn.Module):
    """Small U-Net style CNN for semantic segmentation"""

    def __init__(self, num_classes, num_channels, dropout=False):
        super(CustomCNN, self).__init__()
        self.use_dropout = dropout

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 256 because of skip connection
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 128 because of skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Output layer
        self.dropout = nn.Dropout2d(0.5) if self.use_dropout else nn.Identity()
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)

        # Decoder with skip connections
        up2 = self.upconv2(conv3)
        concat2 = torch.cat([up2, conv2], dim=1)
        conv4 = self.conv4(concat2)

        up1 = self.upconv1(conv4)
        concat1 = torch.cat([up1, conv1], dim=1)
        conv5 = self.conv5(concat1)

        # Output
        x = self.dropout(conv5)
        x = self.out_conv(x)

        return x


def get_network(arch_name: str, num_channels: int, num_classes: int, pretrained: bool, dropout: bool):
    if arch_name == "unet":
        model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet" if pretrained else None, in_channels=num_channels, classes=num_classes)
    elif arch_name == "CustomCNN":
        model = CustomCNN(num_classes=num_classes, num_channels=num_channels, dropout=dropout)
    else:
        raise ValueError(f"Unsupported architecture name: {arch_name}")
    return model
