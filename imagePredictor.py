import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



# === CONFIGURATION ===
IMAGE_PATH = './archive/train/thread_top000.png'
MODEL_PATH = './screw_model.keras'
IMG_SIZE = (256, 256)
# =====================




if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}")
    raise SystemExit(1)

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    raise SystemExit(1)


# Load image for display
image = tf.keras.preprocessing.image.load_img(
    IMAGE_PATH,
    color_mode='rgb',
    target_size=IMG_SIZE
)


# Original array for visualization
image_array = tf.keras.preprocessing.image.img_to_array(image)


# Array for inference
input_array = np.expand_dims(image_array.copy(), axis=0)
input_array = preprocess_input(input_array)

model = tf.keras.models.load_model(MODEL_PATH)

prediction = model.predict(input_array, verbose=0)[0][0]

if prediction >= 0.5:
    tag = 'Bad screw'
    prediction_label = 1
else:
    tag = 'OK screw'
    prediction_label = 0

print(f'Prediction probability: {prediction:.4f}')
print(f'Prediction label: {prediction_label}')
print(f'Result: {tag}')

plt.imshow(image_array.astype(np.uint8))
plt.title(f'{prediction:.4f} => {prediction_label} :: {tag}')
plt.axis('off')
plt.show()