import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the input image
image = cv2.imread('student1.jpeg')  # Replace 'image.jpg' with the filename of your image

# Display the original image
plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.subplot(3, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

# Apply image smoothing using Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)


# Perform edge detection using the Canny edge detection algorithm
edges = cv2.Canny(blurred_image, 100, 200)

# Display the edges
plt.subplot(3, 2, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edges')

####
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Display the image with detected faces
plt.subplot(3,2,4)
plt.imshow(image, cmap='gray')
plt.title('face detected image')

# Show the plots
plt.tight_layout()
plt.show()
