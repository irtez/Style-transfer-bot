import torch.nn as nn
from function import adaptive_instance_normalization as featMod


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groupnum):
        super(ConvLayer, self).__init__()
        # Padding Layer
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groupnum)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        return x

class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, groupnum=1):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)

    def forward(self, x, weight=None, bias=None, filterMod=False):
        if filterMod:
            x1 = self.conv1(x)
            x2 = weight * x1 + bias * x
            
            x3 = self.relu(x2)
            x4 = self.conv2(x3)
            x5 = weight * x4 + bias * x3
            return x + x5
        else: 
            return x + self.conv2(self.relu(self.conv1(x)))

# Control the number of channels
slim_factor = 1

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc1 = nn.Sequential(
            ConvLayer(3, int(16*slim_factor), kernel_size=9, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), int(32*slim_factor), kernel_size=3, stride=2, groupnum=int(16*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(32*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(64*slim_factor), kernel_size=3, stride=2, groupnum=int(32*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ResidualLayer(int(64*slim_factor), kernel_size=3),
            )
        self.enc2 = nn.Sequential(
            ResidualLayer(int(64*slim_factor), kernel_size=3)
            )
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        out = [x1, x2]
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec1 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec2 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(int(64*slim_factor), int(32*slim_factor), kernel_size=3, stride=1, groupnum=int(32*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(32*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(int(32*slim_factor), int(16*slim_factor), kernel_size=3, stride=1, groupnum=int(16*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), int(16*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), 3, kernel_size=9, stride=1, groupnum=1)
            )
        
    def forward(self, x, s, w, b, alpha):
        x1 = featMod(x[1], s[1])
        x1 = alpha * x1 + (1-alpha) * x[1]

        x2 = self.dec1(x1, w[1], b[1], filterMod=True)
        
        x3 = featMod(x2, s[0])
        x3 = alpha * x3 + (1-alpha) * x2
        
        x4 = self.dec2(x3, w[0], b[0], filterMod=True)


        out = self.dec3(x4)
        return out

class Modulator(nn.Module):
    def __init__(self):
        super(Modulator, self).__init__()
        self.weight1 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )  
        self.bias1 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )
        self.weight2 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )  
        self.bias2 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )

    def forward(self, x):
        w1 = self.weight1(x[0])
        b1 = self.bias1(x[0])
        
        w2 = self.weight2(x[1])
        b2 = self.bias2(x[1])
        
        return [w1,w2], [b1,b2]

class TestNet(nn.Module):
    def __init__(self, content_encoder, style_encoder, modulator, decoder):
        super(TestNet, self).__init__()
        
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder


    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        
        style_feats = self.style_encoder(style)
        filter_weights, filter_biases = self.modulator(style_feats)

        content_feats = self.content_encoder(content)
            
        res = self.decoder(content_feats, style_feats, filter_weights, filter_biases, alpha)
        
        return res
