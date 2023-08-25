import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the input image
image = cv2.imread('book.jpeg')  # Replace 'image.jpg' with the filename of your image

# Display the original image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

# Apply image smoothing using Gaussian blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 1)

# Display the blurred image
plt.subplot(2, 2, 3)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')

# Perform edge detection using the Canny edge detection algorithm
edges = cv2.Canny(image, 100, 200)

# Display the edges
plt.subplot(2, 2, 4)
plt.imshow(edges, cmap='gray')
plt.title('Edges')

# Show the plots
plt.tight_layout()
plt.show()
