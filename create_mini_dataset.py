import cv2
import numpy as np
import os

dataset_paths = ['LR_images/nn', 'HR_images/']
output_paths = ['LR_images/nn_mini', 'HR_images_mini/']

num_images = 500

for dataset_path, output_path in zip(dataset_paths, output_paths):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    image_files = []
    
    for file_name in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, file_name)):
            _, ext = os.path.splitext(file_name)
            if ext.lower() in image_extensions:
                image_files.append(os.path.join(dataset_path, file_name))
    
    for i in range(num_images):
        image = cv2.imread(image_files[i])
        cv2.imwrite(os.path.join(output_path, str(i) + '.jpg'), image)