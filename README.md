# Autonomous Driving Simulation using Behavioral Cloning

A deep learning project that trains a Convolutional Neural Network (CNN) to predict steering angles from road images and control a vehicle autonomously in the **Udacity Self-Driving Car Simulator**.

## Overview

This project explores autonomous vehicle control using behavioral cloning, a machine learning approach in which a model learns driving behavior from recorded driving examples.

A CNN processes images captured by the vehicle's front-facing camera and predicts steering angles. The trained model is then connected to the simulator to provide real-time steering and throttle commands.

## Features

- **Behavioral Cloning:** Learns steering behavior from recorded driving data.
- **CNN Architecture:** Uses convolutional layers to extract visual features from road images.
- **Image Preprocessing:** Crops unnecessary image regions, resizes images, and converts them to RGB.
- **Real-Time Predictions:** Generates steering predictions from incoming simulator images.
- **Simulator Integration:** Communicates with the Udacity simulator using Engine.IO.
- **Model Training:** Uses TensorFlow and Keras to train and evaluate the neural network.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development and training |
| OpenCV | Image preprocessing |
| NumPy | Numerical operations and image arrays |
| Pandas | Driving dataset handling |
| Scikit-learn | Training and validation split |
| Pillow | Image processing |
| Engine.IO | Real-time simulator communication |

## Model Architecture

The project uses a CNN inspired by NVIDIA's end-to-end self-driving architecture.

The network consists of:

1. Five convolutional layers for extracting visual features.
2. Flatten layer to convert extracted features into a one-dimensional representation.
3. Dropout layer to help reduce overfitting.
4. Fully connected layers for learning steering behavior.
5. A final output neuron that predicts the steering angle.

**Model configuration:**
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Learning Rate: 0.0001
- Batch Size: 32
- Training Epochs: Up to 20
- Early Stopping and Learning Rate Reduction

## Image Preprocessing

Before images are passed into the CNN, the following preprocessing steps are applied:

- Convert images from BGR to RGB.
- Crop the upper 60 pixels and lower portion of the image to focus on the road.
- Resize images to **200 × 66 pixels**.
- Normalize pixel values inside the model.

This helps the network focus on relevant road information while reducing unnecessary input data.

## Dataset

The model uses driving data recorded through the Udacity Self-Driving Car Simulator.

The dataset contains:
- Center camera images.
- Left and right camera image paths.
- Steering angle measurements.
- Additional driving telemetry recorded by the simulator.

The current baseline training pipeline uses center-camera images and their corresponding steering measurements.

**Note:** The original training dataset is not included in this repository. You need to record your own driving data using the simulator.


## Running Autonomous Driving

1. Start the Udacity Self-Driving Car Simulator.
2. Make sure the trained `model.h5` file is in the project directory.
3. Run the driving script:
4. Wait for the simulator connection.
5. Select **Autonomous Mode** in the simulator.

The model will process incoming camera images and send steering and throttle commands to the simulator.


## Current Limitations

- The vehicle can follow the track autonomously for certain sections but may drift off track during more challenging sections.
- Steering stability and generalization to different road conditions can be improved.
- The model's performance depends on the quality and diversity of the recorded driving data.

## Future Improvements

- Collect more diverse driving data, especially recovery examples.
- Improve steering stability and vehicle control.
- Experiment with data augmentation and model architectures.
- Evaluate performance across different tracks and driving conditions.
- Improve real-time inference and control responsiveness.

## Acknowledgements

- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)
- [NVIDIA End-to-End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)

---

**Developed as a practical exploration of deep learning, computer vision, and autonomous vehicle simulation.**
