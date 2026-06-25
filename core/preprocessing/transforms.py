
from torchvision import transforms


def build_joint_transform(resize=384, crop_size=224):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
