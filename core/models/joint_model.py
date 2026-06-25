
import torch
import torch.nn as nn
import torchvision.models as tv_models


class Identity(nn.Module):
    def forward(self, x):
        return x


class JointModel(nn.Module):
    """
    Transparent reimplementation of AGIN/JOINT scoring model.

    technical branch:
        patch-partitioned image -> Swin-T -> quality_T

    rationality branch:
        LFM image -> ResNet50 -> quality_R

    naturalness:
        0.145 * technical + 0.769 * rationality
    """

    def __init__(self, use_pretrained_backbones=True):
        super().__init__()

        if use_pretrained_backbones:
            swin_weights = tv_models.Swin_T_Weights.DEFAULT
            resnet_weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        else:
            swin_weights = None
            resnet_weights = None

        swin_t = tv_models.swin_t(weights=swin_weights)
        swin_t.head = Identity()

        resnet50 = tv_models.resnet50(weights=resnet_weights)

        self.technical_feature_extraction = swin_t
        self.rationality_feature_extraction = resnet50

        self.quality_T = self.quality_regression(768, 128, 1)
        self.quality_R = self.quality_regression(1000, 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

    def forward(self, x_technical, x_rationality):
        technical_features = self.technical_feature_extraction(x_technical)
        rationality_features = self.rationality_feature_extraction(x_rationality)

        technical_score = self.quality_T(technical_features)
        rationality_score = self.quality_R(rationality_features)

        return technical_score, rationality_score

    @staticmethod
    def naturalness_score(technical_score, rationality_score):
        return 0.145 * technical_score + 0.769 * rationality_score
