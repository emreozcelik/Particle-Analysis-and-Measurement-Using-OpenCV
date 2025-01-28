<p align="center">
  <img src="/process.png">
</p>

# Particle Analysis and Measurement Using OpenCV

### Overview:

This project was created as a tool to assist in writing and researching a master's thesis in mining engineering. The primary goal was to analyze reference images, count particles or fragments, sort them by area, and calculate their real-life dimensions using the known sizes of the reference bars in the images.<br/>

To accomplish this, I developed a solution using Python and OpenCV, which processes the images and performs the necessary calculations.

The solution also includes adding indices to the areas on the image based on their sorted order and generates an Excel file containing the data with columns for index, area, diameter, width, and height, for further analysis.

This process is effective when the analyzed objects are fully contained within the image borders and are of sufficient size.

Overall, this project demonstrates practical applications of image processing techniques in academic research and real-world scenarios.

---

### Code:

#### Image Preview

The image is preprocessed in a photo editor to ensure the reference bars have uniform dimensions before analysis. Also, 475 pixels correspond to 1 meter in real-life measurements.

```python
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
```

<p align="center">
  <img src="/preview.jpg">
</p>

---

#### Processed Image

By processing the image with techniques such as adjusting contrast and applying threshold functions, the image is enhanced, and the blobs are separated. Additionally, to improve detection with adaptiveThreshold, the image is inverted. This method yields very clear results, with most edges distinctly separated rather than blended.

```python
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
```

<p align="center">
  <img src="/enhanced.jpg">
</p>

---

#### Contour Analysis and Data Export

Lastly, the width, height, area, and diameter are calculated using contours. These values are then multiplied by the reference factor to ensure they are represented in meters rather than pixels.

```python
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
```

<p align="center">
  <img src="/final.jpg">
</p>

---

#### Example Data

The table below shows the first 15 values sorted by area as an example. All values in the table are expressed in meters.

<table align="center">
    <tr>
        <td>Index</td>
        <td>Area</td>
        <td>Diameter</td>
        <td>Width</td>
        <td>Height</td>
    </tr>
    <tr>
        <td>984</td>
        <td>0.130100831</td>
        <td>0.4070006423</td>
        <td>0.3721503328</td>
        <td>0.4905357923</td>
    </tr>
    <tr>
        <td>283</td>
        <td>0.1285628809</td>
        <td>0.4045878692</td>
        <td>0.7532537521</td>
        <td>0.286916536</td>
    </tr>
    <tr>
        <td>501</td>
        <td>0.1049041551</td>
        <td>0.3654697234</td>
        <td>0.5166355173</td>
        <td>0.3010320724</td>
    </tr>
    <tr>
        <td>1354</td>
        <td>0.08238891967</td>
        <td>0.323883977</td>
        <td>0.2208919324</td>
        <td>0.8860152395</td>
    </tr>
    <tr>
        <td>444</td>
        <td>0.0720398892</td>
        <td>0.3028597625</td>
        <td>0.5735668303</td>
        <td>0.2748905543</td>
    </tr>
    <tr>
        <td>706</td>
        <td>0.05649196676</td>
        <td>0.2681935981</td>
        <td>0.3949710244</td>
        <td>0.2423498696</td>
    </tr>
    <tr>
        <td>626</td>
        <td>0.05048421053</td>
        <td>0.2535320359</td>
        <td>0.1744685283</td>
        <td>0.4141822253</td>
    </tr>
    <tr>
        <td>882</td>
        <td>0.04984598338</td>
        <td>0.2519243481</td>
        <td>0.2123662206</td>
        <td>0.3003057219</td>
    </tr>
    <tr>
        <td>838</td>
        <td>0.04066481994</td>
        <td>0.2275435273</td>
        <td>0.2042105263</td>
        <td>0.2526315789</td>
    </tr>
    <tr>
        <td>757</td>
        <td>0.04036121884</td>
        <td>0.2266925228</td>
        <td>0.3878727642</td>
        <td>0.1636178749</td>
    </tr>
    <tr>
        <td>1117</td>
        <td>0.03867922438</td>
        <td>0.2219187194</td>
        <td>0.3202889533</td>
        <td>0.1687070345</td>
    </tr>
    <tr>
        <td>902</td>
        <td>0.03826481994</td>
        <td>0.2207267132</td>
        <td>0.2529114894</td>
        <td>0.1987638614</td>
    </tr>
    <tr>
        <td>1075</td>
        <td>0.03729639889</td>
        <td>0.2179156946</td>
        <td>0.279973273</td>
        <td>0.1674944908</td>
    </tr>
    <tr>
        <td>1090</td>
        <td>0.03265373961</td>
        <td>0.2039020171</td>
        <td>0.2973025995</td>
        <td>0.1640528789</td>
    </tr>
    <tr>
        <td>369</td>
        <td>0.03127313019</td>
        <td>0.1995449474</td>
        <td>0.2014967587</td>
        <td>0.214926549</td>
    </tr>
</table>
