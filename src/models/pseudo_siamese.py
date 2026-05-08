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

class PseudoSiameseUNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        
        # ==========================================
        # 1. Encoders
        # ==========================================
        # EO Encoder (3 Channels)
        eo_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.eo_enc0 = nn.Sequential(eo_resnet.conv1, eo_resnet.bn1, eo_resnet.relu) # 64 ch
        self.eo_pool0 = eo_resnet.maxpool
        self.eo_enc1 = eo_resnet.layer1 # 64 ch
        self.eo_enc2 = eo_resnet.layer2 # 128 ch
        self.eo_enc3 = eo_resnet.layer3 # 256 ch
        self.eo_enc4 = eo_resnet.layer4 # 512 ch
        
        # SAR Encoder (1 Channel)
        sar_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify the first conv layer to accept 1 channel instead of 3
        # We reuse the pretrained weights by summing across the RGB channels
        original_conv1 = sar_resnet.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight[:] = torch.sum(original_conv1.weight, dim=1, keepdim=True)
            
        self.sar_enc0 = nn.Sequential(new_conv1, sar_resnet.bn1, sar_resnet.relu) # 64 ch
        self.sar_pool0 = sar_resnet.maxpool
        self.sar_enc1 = sar_resnet.layer1 # 64 ch
        self.sar_enc2 = sar_resnet.layer2 # 128 ch
        self.sar_enc3 = sar_resnet.layer3 # 256 ch
        self.sar_enc4 = sar_resnet.layer4 # 512 ch

        # ==========================================
        # 2. Fusion Decoder
        # ==========================================
        # After fusion (concatenation), all channels are doubled.
        # e4 fused = 512 + 512 = 1024
        # e3 fused = 256 + 256 = 512
        # e2 fused = 128 + 128 = 256
        # e1 fused = 64 + 64 = 128
        # e0 fused = 64 + 64 = 128
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(512 + 512, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(256 + 256, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(128 + 128, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(128 + 128, 128)
        
        # Final upsampling to original resolution
        self.upconv0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec0 = DecoderBlock(64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, eo, sar):
        # -------------------
        # Encoders
        # -------------------
        # EO Path
        eo_e0 = self.eo_enc0(eo)
        eo_p0 = self.eo_pool0(eo_e0)
        eo_e1 = self.eo_enc1(eo_p0)
        eo_e2 = self.eo_enc2(eo_e1)
        eo_e3 = self.eo_enc3(eo_e2)
        eo_e4 = self.eo_enc4(eo_e3)
        
        # SAR Path
        sar_e0 = self.sar_enc0(sar)
        sar_p0 = self.sar_pool0(sar_e0)
        sar_e1 = self.sar_enc1(sar_p0)
        sar_e2 = self.sar_enc2(sar_e1)
        sar_e3 = self.sar_enc3(sar_e2)
        sar_e4 = self.sar_enc4(sar_e3)
        
        # -------------------
        # Fusion
        # -------------------
        f_e0 = torch.cat([eo_e0, sar_e0], dim=1) # 128 ch
        f_e1 = torch.cat([eo_e1, sar_e1], dim=1) # 128 ch
        f_e2 = torch.cat([eo_e2, sar_e2], dim=1) # 256 ch
        f_e3 = torch.cat([eo_e3, sar_e3], dim=1) # 512 ch
        f_e4 = torch.cat([eo_e4, sar_e4], dim=1) # 1024 ch
        
        # -------------------
        # Decoder
        # -------------------
        d4 = self.upconv4(f_e4)
        d4 = torch.cat([d4, f_e3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, f_e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, f_e1], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, f_e0], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.upconv0(d1)
        d0 = self.dec0(d0)
        
        out = self.final_conv(d0)
        return out

if __name__ == "__main__":
    print("Testing PseudoSiameseUNet architecture...")
    model = PseudoSiameseUNet(out_channels=1)
    
    # Dummy inputs: Batch Size = 2
    dummy_eo = torch.randn(2, 3, 256, 256)
    dummy_sar = torch.randn(2, 1, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_eo, dummy_sar)
        
    print(f"EO Input Shape:  {dummy_eo.shape}")
    print(f"SAR Input Shape: {dummy_sar.shape}")
    print(f"Output Shape:    {output.shape}")
    
    assert output.shape == (2, 1, 256, 256), "Output shape mismatch!"
    print("Forward pass successful! Fusion logic works perfectly.")
