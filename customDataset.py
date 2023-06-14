from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, root_low_res, root_high_res, transform=None):
        self.transform = transform
        self.low_res_images = self.get_image_files(root_low_res)
        self.high_res_images = self.get_image_files(root_high_res)

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res_image = Image.open(self.low_res_images[idx])
        high_res_image = Image.open(self.high_res_images[idx])

        if self.transform:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)

        return low_res_image, high_res_image
    
    def get_image_files(self, folder_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
        image_files = []
        
        for file_name in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, file_name)):
                _, ext = os.path.splitext(file_name)
                if ext.lower() in image_extensions:
                    image_files.append(os.path.join(folder_path, file_name))
        
        return image_files
    
