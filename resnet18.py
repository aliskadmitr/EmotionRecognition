import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models 

class ResNet18:
    def __init__(self, num_clusses):
        self.model = models.resnet18(pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_weight()
        self.change_last_layer(num_clusses)
        self.model = self.model.to(self.device)

    def freeze_weight(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_weight(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def change_last_layer(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def train(self, train_loader, val_loader, criterion, optimizer, num_epochs=10):

        for epoch in range(num_epochs):
            self.model.train() 
            running_loss = 0.0
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        
        self.validate(val_loader, criterion)

    def validate(self, val_loader, criterion):
                
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%")



