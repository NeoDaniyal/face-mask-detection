# Real-Time Face Mask Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision system for detecting whether a person is wearing a face mask. The project follows a complete machine learning workflow, starting with a custom CNN baseline and progressing to **ResNet-18 transfer learning**, model evaluation, threshold optimization, and real-time webcam inference using OpenCV.

The final ResNet-18 model achieved **99.54% test accuracy** on 1,088 test images.

---

## Project Overview

The system classifies facial images into two categories:

* `with_mask`
* `without_mask`

The project was developed incrementally to understand how model architecture, data augmentation, transfer learning, and decision-threshold calibration affect classification performance.

### Development Pipeline

```text
Dataset
   ↓
Data Inspection & Cleaning
   ↓
Train / Validation / Test Split
   ↓
Custom CNN Baseline
   ↓
Data Augmentation
   ↓
ResNet-18 Transfer Learning
   ↓
Model Evaluation
   ↓
Threshold Analysis
   ↓
Single-Image Inference
   ↓
Real-Time Webcam Detection
```

---

## Key Features

* **Binary Image Classification**

  * Detects `with_mask` and `without_mask`.

* **Custom CNN Baseline**

  * Four-stage convolutional architecture built with PyTorch.
  * Provides a baseline for comparing more advanced approaches.

* **Transfer Learning**

  * Fine-tuned **ResNet-18** for improved feature extraction and classification performance.

* **Data Augmentation**

  * Applies image transformations during training to improve model generalization.

* **Threshold Analysis**

  * Evaluates different classification thresholds instead of relying exclusively on the default 0.5 probability threshold.
  * An optimized threshold of **0.19** was selected for the evaluated model based on the project's safety-oriented error analysis.

* **Model Evaluation**

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion matrix
  * Prediction/error analysis

* **Single-Image Inference**

  * Run predictions from the command line on individual images.

* **Real-Time Webcam Detection**

  * Uses OpenCV for live camera input.
  * Displays detection results, confidence, FPS, and face-region visualization.

---

## Dataset

The project uses a face-mask image dataset containing two classes:

| Class          | Description                    |
| -------------- | ------------------------------ |
| `with_mask`    | Person wearing a face mask     |
| `without_mask` | Person not wearing a face mask |

The dataset was inspected and cleaned before model training. Exact duplicate images were identified and removed to reduce data leakage between samples.

### Final Dataset Split

| Split      |    Images |
| ---------- | --------: |
| Training   |     5,083 |
| Validation |     1,087 |
| Test       |     1,088 |
| **Total**  | **7,258** |

The test set was kept separate from training and validation during model development.

---

## Model Performance

The project compares three stages of model development:

| Model                           | Test Accuracy | Precision (`without_mask`) | Recall (`without_mask`) |   F1-Score | Errors |
| ------------------------------- | ------------: | -------------------------: | ----------------------: | ---------: | -----: |
| Custom CNN Baseline             |        89.89% |                     90.43% |                  89.78% |     0.9011 |    110 |
| CNN + Data Augmentation         |        90.07% |                     92.61% |                  87.63% |     0.9006 |    108 |
| **ResNet-18 Transfer Learning** |    **99.54%** |                 **99.82%** |              **99.28%** | **0.9955** |  **5** |

The final model reduced classification errors from **110 to 5** on the 1,088-image test set.

---

## ResNet-18 Confusion Matrix

```text
                         Predicted
                    Mask       No Mask
Actual Mask          529          1
Actual No Mask         4        554
```

### Interpretation

* **529** masked faces were correctly classified.
* **554** unmasked faces were correctly classified.
* **1** masked face was classified as unmasked.
* **4** unmasked faces were classified as masked.
* **Total errors: 5 / 1,088**

---

## Project Structure

```text
face-mask-detection/
│
├── config.py
│   └── Project paths and configuration values
│
├── models/
│   └── resnet18_best.pth
│       └── Best ResNet-18 model checkpoint
│
├── outputs/
│   ├── baseline_test_metrics.json
│   ├── resnet18_test_metrics.json
│   ├── test_predictions_resnet18.csv
│   └── plots/
│       ├── accuracy curves
│       ├── confusion matrices
│       └── ROC curves
│
├── src/
│   ├── dataset.py
│   │   └── Custom PyTorch Dataset and image transformations
│   │
│   ├── dataloader.py
│   │   └── Training, validation, and test DataLoaders
│   │
│   ├── model.py
│   │   └── Custom 4-stage CNN baseline
│   │
│   ├── model_transfer.py
│   │   └── ResNet-18 transfer learning architecture
│   │
│   ├── train.py
│   │   └── Custom CNN training pipeline
│   │
│   ├── train_transfer.py
│   │   └── ResNet-18 training pipeline
│   │
│   ├── evaluate_model.py
│   │   └── Test evaluation and prediction analysis
│   │
│   ├── compare_models.py
│   │   └── Model performance comparison
│   │
│   ├── analyze_thresholds.py
│   │   └── Decision-threshold analysis
│   │
│   ├── infer.py
│   │   └── Single-image inference
│   │
│   └── webcam_demo.py
│       └── Real-time OpenCV webcam application
│
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/face-mask-detection.git
cd face-mask-detection
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

Python **3.10+** is recommended.

#### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision opencv-python pillow numpy pandas matplotlib scikit-learn
```

If the project uses additional packages in `dataset.py`, install those as well.

---

## Usage

### Single-Image Inference

Run the trained model on an individual image:

```bash
python src/infer.py --image path/to/sample_image.jpg --threshold 0.19
```

The threshold can be changed to evaluate different decision boundaries.

---

### Real-Time Webcam Detection

Start the OpenCV webcam application:

```bash
python src/webcam_demo.py
```

The application provides a live video feed with detection information.

**Controls:**

* `Q` → Quit
* `ESC` → Quit

A local machine with an accessible webcam is required.

---

## Training

### Train the Custom CNN Baseline

```bash
python src/train.py
```

### Train the ResNet-18 Transfer Learning Model

```bash
python src/train_transfer.py
```

### Evaluate the Model

```bash
python src/evaluate_model.py
```

### Compare Model Versions

```bash
python src/compare_models.py
```

### Analyze Classification Thresholds

```bash
python src/analyze_thresholds.py
```

---

## Training Environment

The project separates development and training workflows:

```text
Local Machine
     ↓
GitHub
     ↓
Google Colab
     ↓
GPU Training
     ↓
Evaluation & Results
     ↓
GitHub
```

Local development is used for writing and testing the Python modules, while Google Colab can be used for GPU-based model training and experimentation.

The final webcam application is intended to run locally because it requires direct access to the computer's webcam.

---

## Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **OpenCV**
* **Pillow**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**

### Machine Learning Concepts

* Convolutional Neural Networks
* Binary Classification
* Data Augmentation
* Transfer Learning
* ResNet-18
* Binary Cross-Entropy with Logits
* Adam Optimization
* Model Checkpointing
* Classification Threshold Optimization
* Confusion Matrix Analysis
* Precision, Recall, and F1-score

---

## Results

The main improvement came from moving beyond the custom CNN baseline to a pretrained ResNet-18 architecture.

```text
Custom CNN
89.89% Accuracy
      ↓
CNN + Augmentation
90.07% Accuracy
      ↓
ResNet-18 Transfer Learning
99.54% Accuracy
```

The final model achieved:

```text
Test Accuracy : 99.54%
F1-Score      : 0.9955
Test Errors   : 5 / 1088
```

---

## Future Improvements

Potential future improvements include:

* Add more diverse real-world face images.
* Evaluate performance under different lighting conditions.
* Test partial/incorrect mask wearing.
* Improve real-time face detection robustness.
* Benchmark inference speed on CPU and GPU.
* Export the model to TorchScript or ONNX.
* Optimize the model for edge/mobile deployment.
* Add automated testing for the inference pipeline.

---

## License

Distributed under the **MIT License**.

See the `LICENSE` file for more information.
