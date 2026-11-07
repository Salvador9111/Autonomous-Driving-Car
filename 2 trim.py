import pandas as pd
import numpy as np
import cv2
import os

# 1. Load your CSV file and tell pandas there are NO headers
# Column 0 = center image, Column 1 = left image, Column 2 = right image, Column 3 = steering
data = pd.read_csv('driving_log.csv', header=None)

# 2. Define a function to preprocess each image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    # Crop the image: remove top 60 pixels (sky) and bottom 25 pixels (hood)
    img = img[60:135, :, :]
    
    # Resize to 66x200 pixels
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    img = img / 255.0 - 0.5
    
    return img

images = []
steering_angles = []

# 3. Loop through rows using numerical positions (Column 0 for center image, Column 3 for steering)
for index, row in data.iterrows():
    img_path_from_csv = row[0] # Column 0 has the center image path
    
    # Extract just the filename (e.g., 'center_2026_...jpg')
    image_filename = os.path.basename(str(img_path_from_csv))
    
    # Combine with your local image folder (change 'IMG' if your folder name is different)
    image_path = os.path.join('IMG', image_filename)
    
    if os.path.exists(image_path):
        processed_img = preprocess_image(image_path)
        images.append(processed_img)
        steering_angles.append(float(row[3])) # Column 3 has the steering angle

# Convert lists to numpy arrays
X_train = np.array(images)
y_train = np.array(steering_angles)

print(f"Successfully processed {len(X_train)} images!")