import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

 
# ==================== CONFIGURATION ====================
EPOCHS = 100             # Higher = more learning, but may lead to overfitting
BATCH_SIZE = 16          # Lower = more stable, higher = faster but potentially less accurate
LEARNING_RATE = 1e-4     # Lower = fine-grained learning, higher = faster but unstable
IMG_SIZE = (256, 256)    # Larger = more detail (better for small defects), uses more memory

# Model
DENSE_UNITS = 128        # Higher = more capacity, but higher risk of overfitting
DROPOUT = 0.3            # Higher = more regularization, prevents overfitting but may slow down learning
FINE_TUNING = False      # True = higher precision (if plenty of data exists), False = more stable/faster

# Data augmentation
AUG_FLIP_H = False       # True = improves generalization if orientation doesn't matter
AUG_FLIP_V = False       # True = useful if parts can be inverted
AUG_ROTATION = 0.05      # Higher = more robust to rotation, too much can cause confusion
AUG_ZOOM = 0.05          # Higher = better with varying distances, too much causes distortion

# Paths
TRAIN_DIR = './archive/train/'      # Folder containing training images
CSV_PATH = './archive/train.csv'    # CSV with labels (0=OK, 1=BAD)
MODEL_PATH = './screw_model.keras'  # Path where the model is saved
# ====================================================== 


# -------------------- Data augmentation --------------------
aug_layers = []
if AUG_FLIP_H:
    aug_layers.append(RandomFlip("horizontal"))
if AUG_FLIP_V:
    aug_layers.append(RandomFlip("vertical"))
if AUG_ROTATION > 0:
    aug_layers.append(RandomRotation(AUG_ROTATION))
if AUG_ZOOM > 0:
    aug_layers.append(RandomZoom(AUG_ZOOM))

data_augmentation = Sequential(aug_layers, name="augmentation") if aug_layers else None


# -------------------- Load data --------------------
data = pd.read_csv(CSV_PATH)

print(f"Total samples: {len(data)}")
print("Class distribution:")
print(data['anomaly'].value_counts().sort_index())

images = []
labels = []

for _, row in data.iterrows():
    img_path = os.path.join(TRAIN_DIR, row['filename'])
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        images.append(np.array(img, dtype=np.float32))
        labels.append(int(row['anomaly']))

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

print(f"Loaded {len(images)} images")


# -------------------- Split --------------------
X_train, X_val, y_train, y_val = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# CORRECT Preprocessing for MobileNetV2
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}")



# -------------------- Class weights --------------------
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
print(f"Class weights: {class_weight_dict}")


# -------------------- Build model --------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = FINE_TUNING

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = inputs

if data_augmentation is not None:
    x = data_augmentation(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = Dense(DENSE_UNITS, activation='relu')(x)
x = Dropout(DROPOUT)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()


# -------------------- Callbacks --------------------
tensorboard_callback = TensorBoard(log_dir="./logs")
early_stopping = EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=5,                 # Stops if no improvement after 5 epochs; allows using high epoch counts without overfitting
    restore_best_weights=True   # Automatically restores the best model found during training
)


# -------------------- Train --------------------
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    callbacks=[tensorboard_callback, early_stopping],
    verbose=1
)


# -------------------- Evaluate --------------------
val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(X_val, y_val, verbose=2)
print(f'Validation accuracy : {val_acc:.4f}')
print(f'Validation precision: {val_precision:.4f}')
print(f'Validation recall   : {val_recall:.4f}')
print(f'Validation AUC      : {val_auc:.4f}')


# -------------------- Debug predictions --------------------
preds = model.predict(X_val, verbose=0).ravel()
pred_labels = (preds >= 0.5).astype(np.int32)

print("Real labels count      :", np.bincount(y_val))
print("Predicted labels count :", np.bincount(pred_labels))
print("Prediction min/max/mean:", preds.min(), preds.max(), preds.mean())


# -------------------- Save --------------------
model.save(MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')