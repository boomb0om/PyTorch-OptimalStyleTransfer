import torch
import torch.nn as nn
import torch.nn.functional as F
from VGGencoder import Encoder
from decoder import Decoder
from feature_transformer import feature_transform


class SingleLevelAE_OST(nn.Module):
    def __init__(self, level, pretrained_path_dir='models'):
        super().__init__()
        self.level = level
        self.encoder = Encoder(f'{pretrained_path_dir}/conv5_1.pth')
        self.decoder = Decoder(level, f'{pretrained_path_dir}/dec{level}_1.pth')

    def forward(self, content_image, style_image, alpha):
        content_feature = self.encoder(content_image, f'relu{self.level}_1')
        style_feature = self.encoder(style_image, f'relu{self.level}_1')
        res = feature_transform(content_feature, style_feature, alpha)
        res = self.decoder(res)
        return res
    

class MultiLevelAE_OST(nn.Module):
    def __init__(self, pretrained_path_dir='models'):
        super().__init__()
        self.encoder = Encoder(f'{pretrained_path_dir}/conv5_1.pth')
        self.decoder1 = Decoder(1, f'{pretrained_path_dir}/dec1_1.pth')
        self.decoder2 = Decoder(2, f'{pretrained_path_dir}/dec2_1.pth')
        self.decoder3 = Decoder(3, f'{pretrained_path_dir}/dec3_1.pth')
        self.decoder4 = Decoder(4, f'{pretrained_path_dir}/dec4_1.pth')
        self.decoder5 = Decoder(5, f'{pretrained_path_dir}/dec5_1.pth')

    def transform_level(self, content_image, style_image, alpha, level):
        content_feature = self.encoder(content_image, f'relu{level}_1')
        style_feature = self.encoder(style_image, f'relu{level}_1')
        res = feature_transform(content_feature, style_feature, alpha)
        return getattr(self, f'decoder{level}')(res)

    def forward(self, content_image, style_image, alpha=1):
        r5 = self.transform_level(content_image, style_image, alpha, 5)
        r4 = self.transform_level(r5, style_image, alpha, 4)
        r3 = self.transform_level(r4, style_image, alpha, 3)
        r2 = self.transform_level(r3, style_image, alpha, 2)
        r1 = self.transform_level(r2, style_image, alpha, 1)

        return r1