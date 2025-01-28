import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive

# Reference: 475 pixels equals 1 meter.
factor = 1 / 475

# Mount Google Drive and read the image.
drive.mount("/content/drive")
img = cv2.imread('/content/drive/MyDrive/img.jpg')

# Calculate the figure size.
img_scale = 0.66
dpi = plt.rcParams['figure.dpi']
height, width, depth = img.shape
figsize = (width * img_scale / float(dpi)), (height * img_scale / float(dpi))

# Display the image.
plt.figure(figsize = figsize)
plt.axis('off')
plt.imshow(img, 'gray')
plt.show()

# Create a copy of the original image and process the image to enhance detection.
blob_img = img.copy()
blob_img = cv2.cvtColor(blob_img, cv2.COLOR_RGB2GRAY)

blob_img = cv2.morphologyEx(blob_img, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
blob_img = cv2.createCLAHE(4, (8, 8)).apply(blob_img)
blob_img = cv2.bitwise_not(blob_img)
blob_img = cv2.GaussianBlur(blob_img, (3, 3), 0)
blob_img = cv2.adaptiveThreshold(blob_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 3)
ret, blob_img = cv2.threshold(blob_img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the image.
plt.figure(figsize = figsize)
plt.axis('off')
plt.imshow(blob_img, 'gray')
plt.show()

# Create a copy of the original image to draw contours on.
contour_img = blob_img.copy()
contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2RGB)
contours, hierarchy = cv2.findContours(blob_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [cnt for cnt in contours if cv2.contourArea(cnt) * factor * factor < 2]

# Define column names for the DataFrame.
columns = ['Index', 'Area', 'Diameter', 'Width', 'Height']
df = pd.DataFrame(columns = columns)

# Process each contour.
for i, cnt in enumerate(contours):
  moments = cv2.moments(cnt)

  if moments["m00"] != 0:
    center, size, angle = cv2.minAreaRect(cnt)

    cX = int(moments['m10'] / moments['m00'])
    cY = int(moments['m01'] / moments['m00'])

    contour_img = cv2.putText(contour_img, str(i), (cX - 16, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    area = cv2.contourArea(cnt) * factor * factor
    diameter = np.sqrt(4 * cv2.contourArea(cnt) / np.pi) * factor
    width = size[0] * factor
    height = size[1] * factor

    df.loc[len(df)] = (i, area, diameter, width, height)

df['Index'] = df['Index'].astype(int)
df = df.sort_values(by = ['Area'], ascending = False)

# Display the image.
plt.figure(figsize = figsize)
plt.axis('off')
plt.imshow(contour_img)
plt.show()

# Save the DataFrame to a CSV file and the image with contours to Google Drive.
df.to_csv('/content/drive/MyDrive/data.csv', index = False)
cv2.imwrite('/content/drive/MyDrive/capture.jpg', cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))

df.head(15).style