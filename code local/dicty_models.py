import torch
from torch import nn

class CNN3DPredictor(nn.Module):
    def __init__(self, K=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
        )
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, K, 1, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, 1, K, H, W)
        x = self.encoder(x)
        x = self.temporal_pool(x)  # (B, 64, 1, H/4, W/4)
        x = x.squeeze(2)  # (B, 64, H/4, W/4)
        x = self.decoder(x)  # (B, 1, H, W)
        return x


class FlowBasedPredictor(nn.Module):
    def __init__(self, K=8):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.combiner = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, K, 1, H, W)
        B, K, C, H, W = x.shape
        last_frame = x[:, -1, :, :, :]
        frame_feat = self.frame_encoder(last_frame)

        if K > 1:
            diffs = []
            for t in range(K-1):
                diff = x[:, t+1, :, :, :] - x[:, t, :, :, :]
                diffs.append(diff)
            flow_input = torch.stack(diffs, dim=1).mean(dim=1)
        else:
            flow_input = torch.zeros_like(last_frame)

        flow_feat = self.flow_encoder(flow_input)
        combined = torch.cat([frame_feat, flow_feat], dim=1)
        features = self.combiner(combined)
        output = self.decoder(features)
        return output


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, hidden):
        h, c = hidden
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMPredictor(nn.Module):
    def __init__(self, K=8, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = ConvLSTMCell(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, K, 1, H, W)
        B, K, C, H, W = x.shape
        h_size = (B, self.hidden_dim, H//4, W//4)
        h = torch.zeros(h_size, device=x.device)
        c = torch.zeros(h_size, device=x.device)
        
        for t in range(K):
            frame = x[:, t, :, :, :]
            features = self.encoder(frame)
            h, c = self.lstm(features, (h, c))
        
        output = self.decoder(h)
        return output


if __name__ == '__main__':
    # Test models
    print("Model Architecture Test")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    K = 8
    batch_size = 2
    H, W = 128, 128
    
    # Create dummy input
    x = torch.randn(batch_size, K, 1, H, W).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Test Model 1: 3D CNN
    print("\n### Model 1: 3D CNN ###")
    model1 = CNN3DPredictor(K=K).to(device)
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"Parameters: {params1:,}")
    
    with torch.no_grad():
        out1 = model1(x)
    print(f"Output shape: {out1.shape}")
    print(f"Output range: [{out1.min():.4f}, {out1.max():.4f}]")
    
    # Test Model 2: Flow-Based
    print("\n### Model 2: Flow-Based ###")
    model2 = FlowBasedPredictor(K=K).to(device)
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"Parameters: {params2:,}")
    
    with torch.no_grad():
        out2 = model2(x)
    print(f"Output shape: {out2.shape}")
    print(f"Output range: [{out2.min():.4f}, {out2.max():.4f}]")
    
    # Test Model 3: ConvLSTM
    print("\n### Model 3: ConvLSTM ###")
    model3 = ConvLSTMPredictor(K=K, hidden_dim=64).to(device)
    params3 = sum(p.numel() for p in model3.parameters())
    print(f"Parameters: {params3:,}")
    
    with torch.no_grad():
        out3 = model3(x)
    print(f"Output shape: {out3.shape}")
    print(f"Output range: [{out3.min():.4f}, {out3.max():.4f}]")
    
    print("\nAll models test passed!")
    print(f"\nModel comparison:")
    print(f"  3D CNN:      {params1:>10,} parameters")
    print(f"  Flow-Based:  {params2:>10,} parameters")
    print(f"  ConvLSTM:    {params3:>10,} parameters")
