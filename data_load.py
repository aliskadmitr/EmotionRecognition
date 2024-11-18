
import os
import torch 
from PIL import Image
import numpy as np
from torch.utils.data import Dataset



class GetData():
    def __init__(self, path):
        self.path = path
    
    def get_files(self):
        data_dir = os.path.join(self.path, 'data')
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                print(f"In '{root}':")
                print("Directories:", dirs)
                print("Files:", files)
                break
        else:
            print("Подкаталог 'data' не найден.")
        return data_dir
    
    def get_paths(self):
        data_dir = self.get_files()
        return self.path, data_dir

class ImageDataset(Dataset):
    def __init__(self, data_dir=None, folder=None, transform=None, images=None, labels=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if images is not None and labels is not None:
            self.image_paths = images
            self.labels = np.array(labels)
        else:
            folder_dir = os.path.join(data_dir, folder)
            for class_folder in os.listdir(folder_dir):
                class_path = os.path.join(folder_dir, class_folder)
                if os.path.isdir(class_path): 
                    for filename in os.listdir(class_path):
                        if filename.endswith(".jpg") or filename.endswith(".png"): 
                            path = os.path.join(class_path, filename)
                            if os.path.isfile(path):
                                self.image_paths.append(path)
                                self.labels.append(int(class_folder))
            self.labels = np.array(self.labels)
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        
        if isinstance(image, str): 
            image = Image.open(image).convert('RGB')
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()
