import random
from torchvision import transforms
from torch.utils.data import DataLoader
from data_load import ImageDataset

def augment_small_classes(dataset, target_classes, target_size, augmentation_transform):
    augmented_images = []
    augmented_labels = []

    for class_label in target_classes:
        class_indices = [i for i, label in enumerate(dataset.labels) if label == class_label]
        
        while len(class_indices) < target_size:
            idx = random.choice(class_indices)
            image, label = dataset[idx]
            augmented_image = augmentation_transform(image)
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
            class_indices.append(idx)

    return augmented_images, augmented_labels

def prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, augmentation_transform, target_classes, target_size):


    augmented_images, augmented_labels = augment_small_classes(
        train_dataset, target_classes, target_size, augmentation_transform
    )

    all_images = train_dataset.image_paths + augmented_images
    all_labels = list(train_dataset.labels) + augmented_labels

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_aug_dataset = ImageDataset(
        data_dir=None, folder=None, images=all_images, labels=all_labels, transform=train_transforms
    )
    val_dataset.transform = test_val_transforms
    test_dataset.transform = test_val_transforms

    train_loader = DataLoader(train_aug_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
