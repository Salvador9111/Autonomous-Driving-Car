
import base64
import json
from io import BytesIO

import cv2
import engineio
import numpy as np

from PIL import Image
from flask import Flask
from tensorflow.keras.models import load_model
from werkzeug.serving import run_simple


# --------------------------------------------------
# 1. Initialize Engine.IO and Flask
# --------------------------------------------------

eio = engineio.Server(
    async_mode='threading',
    cors_allowed_origins='*',
    logger=False
)

flask_app = Flask(__name__)

model = None


# --------------------------------------------------
# 2. Image Preprocessing
# --------------------------------------------------

def preprocess_image(img):

    # Remove sky and car hood
    img = img[60:135, :, :]

    # Resize image to model input dimensions
    img = cv2.resize(
        img,
        (200, 66),
        interpolation=cv2.INTER_AREA
    )

    # Keep raw RGB pixels.
    # The model Lambda layer performs normalization.
    return img.astype(np.uint8)


# --------------------------------------------------
# 3. Send Control Commands
# --------------------------------------------------

def send_control(sid, steering_angle, throttle):

    control_data = {
        'steering_angle': str(float(steering_angle)),
        'throttle': str(float(throttle))
    }

    # Socket.IO EVENT packet
    # 2 = EVENT
    packet = '2' + json.dumps(
        ['steer', control_data],
        separators=(',', ':')
    )

    # Send through Engine.IO connection
    eio.send(sid, packet)


# --------------------------------------------------
# 4. Handle Simulator Connection
# --------------------------------------------------

@eio.on('connect')
def connect(sid, environ):

    print(f'\nSimulator connected! SID: {sid}')


# --------------------------------------------------
# 5. Handle Incoming Telemetry
# --------------------------------------------------

@eio.on('message')
def message(sid, message):

    global model

    try:

        # Check whether model is loaded
        if model is None:
            raise RuntimeError("Model is not loaded!")

        # Convert bytes to string
        if isinstance(message, bytes):
            message = message.decode('utf-8')

        # Process only Socket.IO EVENT packets
        if not isinstance(message, str) or not message.startswith('2'):
            return

        # Decode incoming JSON packet
        packet = json.loads(message[1:])

        if not isinstance(packet, list) or len(packet) < 2:
            return

        event_name = packet[0]
        data = packet[1]

        # Process only telemetry events
        if event_name != 'telemetry':
            return

        if not isinstance(data, dict):
            return
        
        if 'speed' not in data or 'image' not in data:
            return  
            # --------------------------------------------------
        # A. Get Vehicle Speed
        # --------------------------------------------------

        speed = float(data['speed'])

        # --------------------------------------------------
        # B. Decode Camera Image
        # --------------------------------------------------

        img_string = data['image']

        image_bytes = base64.b64decode(img_string)

        image = Image.open(
            BytesIO(image_bytes)
        ).convert('RGB')

        image_np = np.asarray(image)

        # --------------------------------------------------
        # C. Preprocess Image
        # --------------------------------------------------

        processed = preprocess_image(image_np)

        # Add batch dimension
        X = np.expand_dims(processed, axis=0)

        # --------------------------------------------------
        # D. Predict Steering Angle Using CNN
        # --------------------------------------------------

        prediction = model.predict(
            X,
            batch_size=1,
            verbose=0
        )

        steering_angle = float(prediction.flatten()[0])

        # Keep steering angle within training range
        steering_angle = float(
            np.clip(steering_angle, -1.0, 1.0)
        )

        # --------------------------------------------------
        # E. Control Vehicle Speed
        # --------------------------------------------------

        throttle = 0.05 if speed > 15.0 else 0.2

        # --------------------------------------------------
        # F. Send Commands to Simulator
        # --------------------------------------------------

        send_control(
            sid,
            steering_angle,
            throttle
        )

        # --------------------------------------------------
        # G. Display Driving Information
        # --------------------------------------------------

        print(
            f'AUTONOMOUS MODE | '
            f'Steering: {steering_angle:7.4f} | '
            f'Throttle: {throttle:5.2f} | '
            f'Speed: {speed:5.2f}'
        )

    except Exception as error:

        print(f'\nTelemetry error: {error}')

        # Send stop command if an error occurs
        try:
            send_control(sid, 0.0, 0.0)
        except Exception:
            pass


# --------------------------------------------------
# 6. Handle Simulator Disconnection
# --------------------------------------------------

@eio.on('disconnect')
def disconnect(sid):

    print(f'\nSimulator disconnected: {sid}')


# --------------------------------------------------
# 7. Load Model and Start Server
# --------------------------------------------------

if __name__ == '__main__':

    print('Loading model.h5...')

    model = load_model(
        'model.h5',
        safe_mode=False,
        compile=False
    )

    print('Model loaded successfully!')

    # Create Engine.IO WSGI application
    # Simulator connects through /socket.io/
    app = engineio.WSGIApp(
        eio,
        flask_app,
        engineio_path='socket.io'
    )

    print('\nStarting Engine.IO server on port 4567...')
    print('Waiting for simulator connection...')

    run_simple(
        '0.0.0.0',
        4567,
        app,
        threaded=True,
        use_reloader=False
    )
