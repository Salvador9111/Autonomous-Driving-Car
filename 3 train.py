
import os
import math
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Lambda,
    Conv2D,
    Dense,
    Flatten,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

CAMERA_CORRECTION = 0.20
BATCH_SIZE = 32
EPOCHS = 20

print("Loading driving dataset...")


# --------------------------------------------------
# 2. READ CSV FILE
# --------------------------------------------------

data = pd.read_csv("driving_log.csv", header=None)

if data.shape[1] < 4:
    raise ValueError(
        "CSV must contain center, left, right and steering columns."
    )

data["_steering"] = pd.to_numeric(
    data.iloc[:, 3],
    errors="coerce"
)

data = data[data["_steering"].notna()].copy()

if len(data) == 0:
    raise ValueError("No valid steering data found.")


# Split original recording rows before augmentation
train_rows, val_rows = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 3. FIND IMAGE FILES
# --------------------------------------------------

def get_image_path(csv_path):

    filename = os.path.basename(
        str(csv_path).strip().replace("\\", "/")
    )

    return os.path.join("IMG", filename)


# --------------------------------------------------
# 4. CREATE IMAGE RECORDS
# --------------------------------------------------

def create_records(rows, use_all_cameras):

    records = []
    missing_images = 0

    for _, row in rows.iterrows():

        steering = float(row["_steering"])

        if use_all_cameras:

            cameras = [
                (0, 0.0),                    # Center
                (1, CAMERA_CORRECTION),      # Left
                (2, -CAMERA_CORRECTION)      # Right
            ]

        else:

            cameras = [
                (0, 0.0)                     # Center only
            ]

        for column, correction in cameras:

            image_path = get_image_path(row.iloc[column])

            if not os.path.isfile(image_path):
                missing_images += 1
                continue

            adjusted_steering = np.clip(
                steering + correction,
                -1.0,
                1.0
            )

            records.append(
                (image_path, float(adjusted_steering))
            )

    return records, missing_images


train_records, missing_train = create_records(
    train_rows,
    use_all_cameras=True
)

val_records, missing_val = create_records(
    val_rows,
    use_all_cameras=False
)


if len(train_records) == 0 or len(val_records) == 0:
    raise ValueError(
        "No training or validation images found. "
        "Check your IMG folder and CSV paths."
    )


# --------------------------------------------------
# 5. DISPLAY DATASET INFORMATION
# --------------------------------------------------

print("\n========== DATASET REPORT ==========")

print("Original recording rows:", len(data))
print("Training camera images:", len(train_records))
print("Validation center images:", len(val_records))

print("Missing training images:", missing_train)
print("Missing validation images:", missing_val)

print("Training images after flipping:", len(train_records) * 2)

print("====================================\n")


# --------------------------------------------------
# 6. LOAD AND PREPROCESS IMAGES
# --------------------------------------------------

def load_and_preprocess(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not read image: " + image_path)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Remove sky and car hood
    img = img[60:135, :, :]

    if img.shape[0] != 75:
        raise ValueError("Unexpected image size: " + image_path)

    # Resize to 200 x 66
    img = cv2.resize(
        img,
        (200, 66),
        interpolation=cv2.INTER_AREA
    )

    return img.astype(np.uint8)


# --------------------------------------------------
# 7. GENERATE IMAGES WITHOUT USING TOO MUCH RAM
# --------------------------------------------------

def image_generator(records, augment=False):

    for image_path, steering in records:

        img = load_and_preprocess(image_path)

        # Original image
        yield img, np.float32(steering)

        if augment:

            # Flip image horizontally
            flipped_img = cv2.flip(img, 1)

            # Reverse steering direction
            flipped_steering = -steering

            yield flipped_img, np.float32(flipped_steering)


output_signature = (
    tf.TensorSpec(
        shape=(66, 200, 3),
        dtype=tf.uint8
    ),
    tf.TensorSpec(
        shape=(),
        dtype=tf.float32
    )
)


# Training dataset: three cameras + flipped images
train_dataset = tf.data.Dataset.from_generator(
    lambda: image_generator(train_records, augment=True),
    output_signature=output_signature
)

train_dataset = (
    train_dataset
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
    .repeat()
)


# Validation dataset: center camera only
val_dataset = tf.data.Dataset.from_generator(
    lambda: image_generator(val_records, augment=False),
    output_signature=output_signature
)

val_dataset = (
    val_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
    .repeat()
)


# --------------------------------------------------
# 8. BUILD NVIDIA-STYLE CNN MODEL
# --------------------------------------------------

model = Sequential()

# Normalize pixel values
model.add(
    Lambda(
        lambda x: x / 255.0 - 0.5,
        input_shape=(66, 200, 3)
    )
)

# Convolutional layers
model.add(
    Conv2D(24, (5, 5), strides=(2, 2), activation="relu")
)

model.add(
    Conv2D(36, (5, 5), strides=(2, 2), activation="relu")
)

model.add(
    Conv2D(48, (5, 5), strides=(2, 2), activation="relu")
)

model.add(
    Conv2D(64, (3, 3), activation="relu")
)

model.add(
    Conv2D(64, (3, 3), activation="relu")
)

# Fully connected layers
model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))

# Predict steering angle
model.add(Dense(1))


# --------------------------------------------------
# 9. COMPILE MODEL
# --------------------------------------------------

model.compile(
    loss="mse",
    optimizer=Adam(learning_rate=0.0001)
)

model.summary()


# --------------------------------------------------
# 10. TRAIN MODEL
# --------------------------------------------------

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

]

training_samples = len(train_records) * 2

steps_per_epoch = math.ceil(
    training_samples / BATCH_SIZE
)

validation_steps = math.ceil(
    len(val_records) / BATCH_SIZE
)

print("\nStarting model training...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)


# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

model.save("model.h5")

print("\nModel trained successfully!")
print("Saved as model.h5")
