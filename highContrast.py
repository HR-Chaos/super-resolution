import cv2
import numpy as np
import matplotlib.pyplot as plt

def increase_color_contrast(image, factor):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV channels
    h, s, v = cv2.split(hsv)

    # Increase the saturation channel
    s = np.clip(s * factor, 0, 255).astype(np.uint8)

    # Merge the modified channels
    modified_hsv = cv2.merge([h, s, v])

    # Convert the modified image back to the BGR color space
    result = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

    return result

def decrease_color_contrast(image, factor):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV channels
    h, s, v = cv2.split(hsv)

    # Decrease the saturation channel
    s = np.clip(s / factor, 0, 255).astype(np.uint8)

    # Merge the modified channels
    modified_hsv = cv2.merge([h, s, v])

    # Convert the modified image back to the BGR color space
    result = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

    return result

# Load the image
image = cv2.imread('HR_test/pixabay_dog_001363.jpg')

# Increase the color contrast (adjust the factor as desired)
contrast_image = increase_color_contrast(image, factor=1.5)

# Revert the contrast enhancement (adjust the factor accordingly)
reversed_image = decrease_color_contrast(contrast_image, factor=1.5)

# Display the images in the terminal
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Contrast-enhanced')
axes[1].axis('off')
axes[2].imshow(cv2.cvtColor(reversed_image, cv2.COLOR_BGR2RGB))
axes[2].set_title('Reversed')
axes[2].axis('off')
plt.show()