import torch
import torch.nn as nn

class AdaptiveScale(nn.Module):
    """可学习缩放因子模块"""
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))  # 可学习缩放参数

    def forward(self, x):
        return x * self.scale  # 对输入特征进行通道级缩放

class DeepConv(nn.Module):
    """深度特征提取模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 深度特征提取路径
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2*in_channels, 4*in_channels, 1)
        )
        # 残差路径
        self.conv2 = nn.Sequential(
            nn.Conv2d(4*in_channels, 4*in_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4*in_channels, out_channels, 1)
        )
、      self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity

class OutConv(nn.Module):
    """输出卷积模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, 1)
        )
    def forward(self,x):
      return self.conv(x)


# ------------------- 通道注意力机制 -------------------
class ChannelAttention(nn.Module):
    """通道注意力模块（SE注意力变体）
    通过同时考虑平均池化和最大池化特征，增强重要通道的响应
    Args:
        in_planes (int): 输入特征图的通道数

    Forward:
        Input: [B, C, H, W]
        Output: [B, C, 1, 1] 通道注意力权重
    """

    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        # 双路池化：平均池化+最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 瓶颈结构全连接层
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)  # 降维
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)  # 升维
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 双路特征提取
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 特征融合与激活
        out = avg_out + max_out  # 逐元素相加
        return self.sigmoid(out)  # 压缩到0-1范围


# ------------------- 物理感知编码模块 -------------------
class PhysicsAwareEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 空间特征提取
            nn.GELU(),  # 平滑激活函数
            WindPhysicsBlock(out_channels)  # 嵌入物理约束模块
        )

    def forward(self, x):
        return self.conv(x)


class WindPhysicsBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        # 物理过程建模层（5x5卷积模拟局部风应力效应）
        self.phys_layer = nn.Conv2d(channels, channels, 5, padding=2)
        # 特征适配器（1x1卷积融合原始特征和物理项）
        self.adaptor = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        # 物理驱动项建模：近似风速平方效应
        physics_term = torch.square(x)  # U^2项
        physics_term = self.phys_layer(physics_term)  # 空间相关性建模
        # 特征融合：原始特征+物理修正项
        return self.adaptor(torch.cat([x, physics_term], dim=1))


# ------------------- 跨模态注意力机制 -------------------
class CrossAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        # 查询、键、值映射
        self.query = nn.Conv2d(channels, channels // 4, 1)  # 降维减少计算量
        self.key = nn.Conv2d(channels, channels // 4, 1)
        self.value = nn.Conv2d(channels, channels // 4, 1)

        # 自适应缩放因子（可学习参数）
        self.scale = AdaptiveScale(channels // 4)
        self.out_conv = nn.Conv2d(channels // 4, channels, 1)  # 恢复通道数

    def forward(self, x, context):
        # 计算注意力权重
        Q = self.query(x)
        K = self.key(context)
        V = self.value(context)
        att = self.scale(torch.softmax(Q @ K, dim=1))  # 空间维度softmax
        return self.out_conv(att * V)  # 注意力加权后的特征


class DynamicFusionBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        # 双向注意力机制
        self.sat_att = CrossAttention(channels)  # 卫星关注风场
        self.wind_att = CrossAttention(channels)  # 风场关注卫星
        # 特征融合卷积
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, sat, wind):
        # 交叉注意力特征增强
        sat_weighted = self.sat_att(sat, wind)  # 卫星特征中融入风场信息
        wind_weighted = self.wind_att(wind, sat)  # 风场特征中融入卫星信息

        # 残差连接与特征拼接
        fused = self.fusion_conv(torch.cat([
            sat + sat_weighted,  # 增强后的卫星特征
            wind + wind_weighted  # 增强后的风场特征
        ], dim=1))
        return fused


# ------------------- 双通道物理交互模块 -------------------
class DC_PIM(nn.Module):

    def __init__(self, sat_channels, wind_channels, hidden_dim=64):
        super().__init__()
        # 卫星测量编码器（含通道注意力）
        self.sat_encoder = nn.Sequential(
            nn.Conv2d(sat_channels, hidden_dim, 3, padding=1),  # 空间特征提取
            nn.GELU(),  # 平滑非线性激活
            ChannelAttention(hidden_dim),  # 通道维度特征选择
        )
        # 物理场编码器（嵌入物理约束）
        self.wind_encoder = PhysicsAwareEncoder(wind_channels, hidden_dim)
        # 动态特征融合模块
        self.fusion = DynamicFusionBlock(hidden_dim)

    def forward(self, sat, wind):
        # 特征编码
        sat_feat = self.sat_encoder(sat)  # [B,64,H,W]
        wind_feat = self.wind_encoder(wind)  # [B,64,H,W]
        # 跨模态特征融合
        fused = self.fusion(sat_feat, wind_feat)
        return fused


# ------------------- 基础网络组件 -------------------
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.2),  # 负斜率0.2的LeakyReLU
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear'),  # 双线性下采样
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Upm(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class SAH(nn.Module):
    def __init__(self, x_in,y_in, out_channels):
        super().__init__()
        # 待补充
        self.conv1 = nn.Conv2d(x_in,x_in,3)
    def forward(self, x,y):
        x0 = x
        x = self.conv1(x)
        return x

# ------------------- 主网络 -------------------
class SSHNET(nn.Module):
    def __init__(self,x_in,y_in,z_in,out_ch):
        super(SSHNET,self).__init__()
        self.convy = DeepConv(y_in,y_in) #y 为全球海面风场和大气层数据,包括wind_speed等
        self.convz = DeepConv(z_in, z_in) # z 为降雨，海况等环境数据,包括降雨标记，降雪标记，海况类型等
        self.convx = DeepConv(x_in,out_ch) # x 为卫星测量数据，包括Range,ALT,SWH等
        self.DC_PIM1 = DC_PIM(x_in,y_in,y_in) # 注意力机制 卫星测量数据对于海面风场，大气层数据的注意力机制
        self.DC_PIM2 = DC_PIM(x_in,z_in,z_in) #  卫星测量对于降雨，海况等环境数据的注意力机制
        self.sch1 = SAH(x_in,y_in,x_in+y_in) # 对x，y 进行交互
        self.convs1 = nn.Conv2d(x_in+y_in,(x_in+y_in)//2,kernel_size=3,stride=1,padding=1,bias=True)
        self.convs2 = nn.Conv2d(x_in + z_in, (x_in + z_in) // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.sch2 = SAH(x_in,z_in,x_in+z_in) # x和z进行交互
        self.DC_PIM3 = DC_PIM((x_in+y_in)//2,(x_in+z_in)//2,(x_in+y_in+z_in)//3)
        self.sch3 =  SAH(x_in,(x_in+y_in+z_in)//3,(x_in+y_in+z_in)//3)
        self.out = OutConv((x_in+y_in+z_in)//3,out_ch)
        self.ca = ChannelAttention((x_in+z_in+y_in)//3)
        self.conv = nn.Sequential(
            DoubleConv(x_in+z_in+y_in,(x_in+z_in+y_in)//3),
            DeepConv((x_in+z_in+y_in)//3,(x_in+z_in+y_in)//3)
        )
        self.convs3 = DoubleConv((x_in+y_in+z_in)//3, (x_in+y_in+z_in)//3)
    def forward(self,x,y,z):
        y1 = self.convy(y)
        z1 = self.convz(z)
        x1 = self.DC_PIM1(x,y1)
        x2 = self.DC_PIM2(x,z1)
        y1 = self.convy(x1+y1)
        z1 = self.convz(x2 + z1)
        xh1 = self.convs1(self.sch1(x,y1))
        xh2 = self.convs2(self.sch2(x,z1))
        dc1 = self.DC_PIM3(xh1,xh2)
        xh3 = self.convs3(self.sch3(x,dc1))
        t = torch.cat((x,y,z),dim=1)
        t = self.conv(t)
        t = self.convs3(self.ca(t)*t)
        out = self.out(t*xh3)
        return out + self.convx(x)
