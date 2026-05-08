import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class ResNet18UNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        
        # Encoder: Pretrained ResNet-18
        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Extract encoder layers
        self.enc0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu) # 64 channels, 1/2 resolution
        self.pool0 = encoder.maxpool # 1/4 resolution
        self.enc1 = encoder.layer1 # 64 channels, 1/4 resolution
        self.enc2 = encoder.layer2 # 128 channels, 1/8 resolution
        self.enc3 = encoder.layer3 # 256 channels, 1/16 resolution
        self.enc4 = encoder.layer4 # 512 channels, 1/32 resolution

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(256 + 256, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(128 + 128, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(64 + 64, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(64 + 64, 64)
        
        # Final upsampling to original resolution (1/2 -> 1/1)
        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = DecoderBlock(32, 32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        p0 = self.pool0(e0)
        e1 = self.enc1(p0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Decoder
        d4 = self.upconv4(e4)
        d4 = torch.cat([d4, e3], dim=1) # Skip connection
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1) # Skip connection
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1) # Skip connection
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e0], dim=1) # Skip connection
        d1 = self.dec1(d1)
        
        d0 = self.upconv0(d1)
        d0 = self.dec0(d0)
        
        out = self.final_conv(d0)
        return out

if __name__ == "__main__":
    print("Testing ResNet18UNet architecture...")
    model = ResNet18UNet(out_channels=1)
    
    # Dummy input: Batch Size = 2, Channels = 3 (EO), Height = 256, Width = 256
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == (2, 1, 256, 256), "Output shape mismatch!"
    print("Forward pass successful!")
