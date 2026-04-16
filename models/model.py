import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn = SpatialAttention2D(out_ch)

    def forward(self, x):
        x = self.block(x)
        x = self.attn(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.block(x)
        return x


class AttentionAutoencoder2D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, latent_channels=256):
        super().__init__()

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16
        self.enc1 = EncoderBlock(in_channels, base_filters, stride=2)       # -> 128
        self.enc2 = EncoderBlock(base_filters, base_filters * 2, stride=2)  # -> 64
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4, stride=2)  # -> 32
        self.enc4 = EncoderBlock(base_filters * 4, latent_channels, stride=2)   # -> 16

        # Bottleneck attention
        self.bottleneck_attn = SpatialAttention2D(latent_channels)

        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
        self.dec4 = DecoderBlock(latent_channels, base_filters * 4)
        self.dec3 = DecoderBlock(base_filters * 4, base_filters * 2)
        self.dec2 = DecoderBlock(base_filters * 2, base_filters)
        self.dec1 = DecoderBlock(base_filters, base_filters)

        self.final = nn.Sequential(
            nn.Conv2d(base_filters, in_channels, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_attn(x)
        return x

    def decode(self, z):
        z = self.dec4(z)
        z = self.dec3(z)
        z = self.dec2(z)
        z = self.dec1(z)
        return self.final(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def build_model(device):
    model = AttentionAutoencoder2D(in_channels=1, base_filters=32, latent_channels=256)
    model = model.to(device)
    return model
