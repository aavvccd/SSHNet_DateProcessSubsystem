import torch
import torch.nn as nn

class AdaptiveScale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x * self.scale

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)  # 降维
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)  # 升维
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x


class CrossAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 4, 1)
        self.key = nn.Conv2d(channels, channels // 4, 1)
        self.value = nn.Conv2d(channels, channels // 4, 1)
        self.scale = AdaptiveScale(channels // 4)
        self.out_conv = nn.Conv2d(channels // 4, channels, 1)

    def forward(self, x, context):
        Q = self.query(x)
        K = self.key(context)
        V = self.value(context)
        att = self.scale(torch.softmax(Q * K, dim=1))
        return self.out_conv(att * V)
class DynamicFusionBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.sat_att = CrossAttention(channels)
        self.wind_att = CrossAttention(channels)

        self.fusion_conv = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, sat, wind):

        sat_weighted = self.sat_att(sat, wind)
        wind_weighted = self.wind_att(wind, sat)

        fused = self.fusion_conv(torch.cat([
            sat + sat_weighted,
            wind + wind_weighted
        ], dim=1))
        return fused

class DeepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2*in_channels, 4*in_channels, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4*in_channels, 4*in_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4*in_channels, out_channels, 1)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity
class PhysicsAwareEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 空间特征提取
            nn.GELU(),
            WindPhysicsBlock(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class WindPhysicsBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.phys_layer = nn.Conv2d(channels, channels, 5, padding=2)
        self.adaptor = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        physics_term = torch.square(x)
        physics_term = self.phys_layer(physics_term)
        return self.adaptor(torch.cat([x, physics_term], dim=1))
class SSHCompensationNet(nn.Module):
    def __init__(self, sat_channels=15, env_channels=3, out_channels=1):
        super().__init__()

        self.sat_encoder = nn.Sequential(
            nn.Conv2d(sat_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            ChannelAttention(64),
            DeepConv(64, 64)
        )

        self.env_encoder = PhysicsAwareEncoder(env_channels, 64)

        self.interaction = DynamicFusionBlock(64)
        self.feature_enhancer = nn.Sequential(
            DeepConv(64, 64),
            ChannelAttention(64)
        )
        self.compensation_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, out_channels, 1)
        )
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, sat_measurements, env_features):
        sat_feat = self.sat_encoder(sat_measurements)
        env_feat = self.env_encoder(env_features)
        interacted = self.interaction(sat_feat, env_feat)
        enhanced = self.feature_enhancer(interacted)
        compensation = self.compensation_head(enhanced)
        return compensation
