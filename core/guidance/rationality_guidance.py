
import torch
import torch.nn as nn
import torch.nn.functional as F


class RationalityGuidanceScorer(nn.Module):
    """
    Differentiable rationality scorer for inference guidance.

    Input:
        decoded image tensor in [0, 1], shape [B, 3, H, W]

    Pipeline:
        image -> low-pass blur -> resize 224 -> ImageNet normalize
              -> JOINT ResNet50 rationality branch -> quality_R score
    """

    def __init__(self, joint_model, image_size=224, blur_kernel=11):
        super().__init__()

        self.joint_model = joint_model
        self.image_size = image_size
        self.blur_kernel = blur_kernel

        for p in self.joint_model.parameters():
            p.requires_grad_(False)

        self.joint_model.eval()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def low_pass(self, x):
        return F.avg_pool2d(
            x,
            kernel_size=self.blur_kernel,
            stride=1,
            padding=self.blur_kernel // 2,
        )

    def preprocess(self, x):
        x = self.low_pass(x)

        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        x = (x - self.mean) / self.std
        return x

    def forward(self, x):
        x = self.preprocess(x)
        features = self.joint_model.rationality_feature_extraction(x)
        score = self.joint_model.quality_R(features)
        return score
