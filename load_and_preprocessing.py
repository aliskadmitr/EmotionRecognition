# import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split


# parameters
BATCH_SIZE = 64
IMG_SIZE = 128
NUM_CLASSES = 7
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

print('======================Загрузка началась====================')

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # Преобразование в диапазон [-1, 1] для одноканальных изображений
    transforms.Normalize(mean=[0.5], std=[0.5])
])


train_dataset = datasets.ImageFolder(root='train', transform=transform)
test_dataset = datasets.ImageFolder(root='test', transform=transform)


train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size])

# загружаем в загрузчик данных
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('======================Загрузка прошла успешно====================')
