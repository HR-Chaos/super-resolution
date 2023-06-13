import cv2
import numpy as np
import os

#################### For saving all in the same folder ####################
# Define the paths, change at convenience
input_folder = 'Downscale_Testing'
output_folder = 'Downscale_Target'

nn_folder = output_folder
bi_folder = output_folder
ci_folder = output_folder
ar_folder = output_folder

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

#################### For separate folders based on method ####################
# nn_folder = 'define_here'
# bi_folder = 'define_here'
# ci_folder = 'define_here'
# ar_folder = 'define_here'



# Get a list of all image files in the input folder
image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Iterate over each image file
for image_file in image_files:
    # Load the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    #################### Define desired output dimensions here ####################
    width = 64
    height = 64

    ################ Nearest Neighbor ################
    downscaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    # Save the downscaled image
    output_path = os.path.join(nn_folder, f"downscaled_nn_{image_file }")
    cv2.imwrite(output_path, downscaled_image)
    
    ################ Bilinear Interpolation ################
    downscaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    # Save the downscaled image
    output_path = os.path.join(bi_folder, f"downscaled_bi_{image_file}")
    cv2.imwrite(output_path, downscaled_image)
    
    ################ Cubic Interpolation ################
    downscaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    # Save the downscaled image
    output_path = os.path.join(ci_folder, f"downscaled_ci_{image_file}")
    cv2.imwrite(output_path, downscaled_image)
    
    ################ Area Based Resampling ################
    downscaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    # Save the downscaled image
    output_path = os.path.join(ar_folder, f"downscaled_ar_{image_file}")
    cv2.imwrite(output_path, downscaled_image)

print("Image downscaling completed.")