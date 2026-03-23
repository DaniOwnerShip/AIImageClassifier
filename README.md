# 🔩 Screw Quality Inspection (OK vs Defective)

This project explores the use of deep learning for **visual quality inspection** in an industrial-like scenario.
The goal is to automatically classify screws as:

* **OK (normal)**
* **Defective (anomalous)**

Although not my core OT/IT niche, this project serves as a **hands-on exploration of computer vision applied to manufacturing and inspection workflows**.

---

## 📌 Overview

The model is based on **transfer learning** using a pre-trained convolutional neural network.
Instead of training from scratch, a model already trained on millions of images is adapted to detect defects in screws.

This approach allows:

* Faster development
* Better performance with limited data
* Practical applicability in industrial contexts

---

## 🧠 Model Architecture

The solution uses:

* **MobileNetV2 (pre-trained on ImageNet)** as a feature extractor
* A custom classification head:

  * Global Average Pooling
  * Dense layer (ReLU)
  * Dropout (regularization)
  * Sigmoid output (binary classification)

Pipeline:

```
Image → MobileNetV2 → Feature extraction → Classifier → OK / Defective
```

---

## ⚙️ Key Features

* Binary classification (OK vs Defective)
* Transfer learning with pre-trained CNN
* Class imbalance handling via `class_weight`
* Data augmentation (rotation, zoom, flips)
* Early stopping based on validation AUC
* Multiple evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * AUC

---

## 📂 Project Structure

```
.
├── archive/
│   ├── train/           # Training images
│   └── train.csv        # Labels (0 = OK, 1 = Defective)
├── screw_model.keras    # Trained model
├── train.py             # Training script
├── predict.py           # Inference script
└── logs/                # TensorBoard logs
```

---

## 🚀 Training

Run:

```bash
python train.py
```

The training pipeline includes:

* Image loading and preprocessing
* Train/validation split
* Class weight balancing
* Model training with early stopping

---

## 🔍 Inference

To test a single image:

```bash
python predict.py
```

Output example:

```
Prediction probability: 0.8732
Prediction label: 1
Result: Bad screw
```

---

## ⚠️ Notes & Considerations

* Input images are preprocessed using **MobileNetV2 preprocessing** (range [-1, 1])
* Training and inference pipelines are fully aligned
* Dataset quality has a strong impact on performance:

  * Lighting consistency
  * Framing of the screw
  * Clear defect visibility

---

## 📊 Typical Challenges

* Class imbalance (more OK than defective samples)
* Subtle defects (scratches, deformation)
* Variability in acquisition conditions

---

## 🏭 Industrial Context

This type of solution can be applied to:

* Automated quality inspection
* Inline defect detection
* Visual validation in manufacturing lines
* OT/IT integration scenarios (edge AI, vision systems)

---

## 🧪 Purpose of This Project

This project is part of my **Lab / Exploration work** and aims to:

* Understand practical ML deployment challenges
* Explore computer vision in industrial environments
* Bridge OT (operations) and IT (data/AI systems)

---

## 🔮 Possible Improvements

* Fine-tuning the base model (MobileNetV2)
* Higher resolution input images
* Better dataset curation
* Confusion matrix & error analysis
* Deployment on edge devices (Raspberry Pi / industrial PC)

---

## 📎 Tech Stack

* Python
* TensorFlow / Keras
* NumPy / Pandas
* scikit-learn
* PIL / Matplotlib

---

## 👤 Author

Part of my **OT/IT Labs portfolio**, exploring intersections between:

* Industrial systems
* Data-driven approaches
* Machine learning applications

---
