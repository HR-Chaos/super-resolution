import os
import random
import shutil


####################################################################
#Important: Files remain in the same order among HR LR folder pairs
####################################################################

# Define the paths - default only applicable after running downscale.ipynb
hr_images_dir = 'HR_images'

#Modify ci for whatever method is in use
lr_images_dir = 'LR_images' + '/' + "ci"

hr_train_dir = 'HR_train'
lr_train_dir = 'LR_train'
hr_val_dir = 'HR_val'
lr_val_dir = 'LR_val'
hr_test_dir = 'HR_test'
lr_test_dir = 'LR_test'

# Create the destination directories if they don't exist
os.makedirs(hr_train_dir, exist_ok=True)
os.makedirs(lr_train_dir, exist_ok=True)
os.makedirs(hr_val_dir, exist_ok=True)
os.makedirs(lr_val_dir, exist_ok=True)
os.makedirs(hr_test_dir, exist_ok=True)
os.makedirs(lr_test_dir, exist_ok=True)

# Get a list of all HR image files in the HR_images directory
hr_image_files = [f for f in os.listdir(hr_images_dir) if os.path.isfile(os.path.join(hr_images_dir, f))]

# Shuffle the HR image files
random.shuffle(hr_image_files)

# Move HR and LR image pairs to the corresponding directories
train_files = hr_image_files[:100]

for file_name in train_files:
    hr_src_path = os.path.join(hr_images_dir, file_name)
    hr_dst_path = os.path.join(hr_train_dir, file_name)
    lr_file_name = f"downscaled_ci_{file_name}"
    lr_src_path = os.path.join(lr_images_dir, lr_file_name)
    lr_dst_path = os.path.join(lr_train_dir, lr_file_name)

    shutil.move(hr_src_path, hr_dst_path)
    shutil.move(lr_src_path, lr_dst_path)

val_files = hr_image_files[100:150]

for file_name in val_files:
    hr_src_path = os.path.join(hr_images_dir, file_name)
    hr_dst_path = os.path.join(hr_val_dir, file_name)
    lr_file_name = f"downscaled_ci_{file_name}"
    lr_src_path = os.path.join(lr_images_dir, lr_file_name)
    lr_dst_path = os.path.join(lr_val_dir, lr_file_name)

    shutil.move(hr_src_path, hr_dst_path)
    shutil.move(lr_src_path, lr_dst_path)

for file_name in hr_image_files[150:170]:
    hr_src_path = os.path.join(hr_images_dir, file_name)
    hr_dst_path = os.path.join(hr_test_dir, file_name)
    lr_file_name = f"downscaled_ci_{file_name}"
    lr_src_path = os.path.join(lr_images_dir, lr_file_name)
    lr_dst_path = os.path.join(lr_test_dir, lr_file_name)

    shutil.move(hr_src_path, hr_dst_path)
    shutil.move(lr_src_path, lr_dst_path)

print("Files moved successfully!")