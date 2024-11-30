from torchvision.transforms import (
    Compose, Resize, RandomRotation, RandomAdjustSharpness,
    RandomHorizontalFlip, Normalize, ToTensor
)
from transformers import AutoImageProcessor

class ImageTransformations:
    def __init__(self, model_checkpoint: str):
        self.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
        self.image_mean, self.image_std = self.image_processor.image_mean, self.image_processor.image_std
        self.size = self.image_processor.size["height"]

    def get_train_transforms(self):
        return Compose([
            Resize((self.size, self.size)),
            RandomRotation(90),
            RandomAdjustSharpness(2),
            RandomHorizontalFlip(0.5),
            ToTensor(),
            Normalize(mean=self.image_mean, std=self.image_std)
        ])

    def get_val_transforms(self):
        return Compose([
            Resize((self.size, self.size)),
            ToTensor(),
            Normalize(mean=self.image_mean, std=self.image_std)
        ])
