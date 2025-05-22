import torch
import timm
from torch import nn
import segmentation_models_pytorch as smp


class CustomCNN(nn.Module):
    def __init__(self, num_classes, num_channels, dropout=False):
        super(CustomCNN, self).__init__()
        self.use_dropout = dropout

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout() if self.use_dropout else nn.Identity(),  # Dropout before FC layer
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_network(arch_name: str, num_channels: int, num_classes: int, pretrained: bool, dropout: bool):
    torch.manual_seed(42)
    drop_rate = 0.5 if dropout else 0.0
    if arch_name == "ResNet18":
        model = timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes, in_chans=num_channels, drop_rate=drop_rate)
    elif arch_name == "unet":
        model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet" if pretrained else None, in_channels=num_channels, classes=num_classes)
    elif arch_name == "CustomCNN":
        model = CustomCNN(num_classes=num_classes, num_channels=num_channels, dropout=dropout)
    elif arch_name == "ConvNeXt-Nano":
        model = timm.create_model("convnext_nano", pretrained=pretrained, num_classes=num_classes, in_chans=num_channels, drop_rate=drop_rate)
    elif arch_name == "ViT-Tiny":
        model = timm.create_model(
            'vit_tiny_patch16_224',
            patch_size=8,
            img_size=112,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=num_channels,
            drop_rate=drop_rate
        )
    else:
        raise ValueError(f"Unsupported architecture name: {arch_name}")
    return model
