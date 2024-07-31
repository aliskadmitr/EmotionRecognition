from load_and_preprocessing import *
import torch
import torch.nn as nn
from tqdm import tqdm


KERNEL_SIZE = 3


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(16 * 16 * 128, 1000)
        self.fc2 = nn.Linear(1000, 7)
# определяем потоки данных через слои

    def forward(self, x):
        out = self.layer1(x)
        # print(f"After layer1: {out.shape}")
        out = self.layer2(out)
        # print(f"After layer2: {out.shape}")
        out = self.layer3(out)
        # print(f"After layer3: {out.shape}")
        out = out.reshape(out.size(0), -1)
        # print(f"After reshaping: {out.shape}")
        out = self.drop_out(out)
        out = self.fc1(out)
        # print(f"After fc1: {out.shape}")
        out = self.fc2(out)
        # print(f"After fc2: {out.shape}")
        return out


model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Устройство (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if device.type == 'cuda':
    print("Запуск на GPU")
    print(torch.cuda.get_device_name(0))  # Вывод имени GPU
else:
    print("Запуск на CPU")


total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        images, labels = images.to(device), labels.to(device)

        # Прямой запуск
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Обратное распространение и оптимизатор
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Отслеживание точности
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

print("================Начало тестирования=================")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        (correct / total) * 100))


print("================конец тестирования=================")


# график
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
