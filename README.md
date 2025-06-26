# Deep Learning Applications: Medical Image Classification & Sentiment Analysis

## 1. Introduction

This project explores two fundamental applications of deep learning:

- **Medical Image Classification**: Classifying chest X-ray images into "Pneumonia" or "Normal" using CNNs and transfer learning.
- **Sentiment Analysis**: Classifying IMDb reviews as positive or negative using NLP techniques.

The objective is to evaluate different neural network architectures and compare their performance on both tasks, focusing on real-world applicability.

---

## 2. Dataset Preparation

### 2.1 Medical Image Classification

- **Dataset**: [Chest X-ray Images (Pneumonia) by Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total images**: 5,863  
- **Classes**: Pneumonia, Normal  
- **Split**:
  - Training: 5,216 images
  - Validation: 16 images
  - Testing: 624 images

#### Preprocessing:
- Resized all images to **150x150 pixels**
- Converted to **grayscale**
- Normalized pixel values to **[0, 1]**
- Applied **data augmentation**:
  - Rotation (±30°)
  - Zoom (up to 20%)
  - Horizontal & vertical shifts (±10%)
  - Horizontal flipping

### 2.2 Sentiment Analysis

- **Dataset**: IMDb Movie Reviews
- **Total reviews**: 50,000  
- **Classes**: Positive, Negative  
- **Split**:
  - Training: 25,000
  - Testing: 25,000

#### Preprocessing:
- Tokenized reviews into integer sequences
- Padded sequences to **200 words**
- Vocabulary size limited to **top 10,000** frequent words

---

## 3. Model Creation & Training

### 3.1 Medical Image Classification

#### 🔹 Model 1: Custom CNN
- Convolutional layers with 32, 64, 128, 256 filters
- Batch normalization & dropout
- Fully connected dense layers
- Final activation: **Sigmoid**

#### 🔹 Model 2: Transfer Learning with MobileNetV2
- Pretrained on ImageNet, base layers frozen
- Global average pooling + dense layers
- Dropout for regularization
- Fine-tuned by unfreezing layers after the 100th

### 3.2 Sentiment Analysis

#### 🔹 Model 1: Feedforward Neural Network
- Embedding layer (128 dimensions)
- Flatten + 2 dense layers (ReLU)
- Final activation: **Sigmoid**

#### 🔹 Model 2: LSTM-based Network
- Embedding layer (64 dimensions)
- Bidirectional LSTM (64 units) + LSTM (32 units)
- Dropout + Dense layer (Sigmoid)

---

## 4. Hyperparameters & Training

### Medical Image Classification

| Model        | Optimizer | LR     | Epochs | Batch Size |
|--------------|-----------|--------|--------|-------------|
| Custom CNN   | Adam      | 0.0003 | 12     | 32          |
| MobileNetV2  | Adam      | 0.0001 | 10 + 15 (fine-tuning) | 32 |

### Sentiment Analysis

| Model                  | Optimizer | Epochs | Batch Size |
|------------------------|-----------|--------|-------------|
| Feedforward NN, LSTM   | Adam      | 5      | 32          |

---

## 5. Evaluation & Results

### 5.1 Medical Image Classification

#### ✅ Custom CNN
- **Accuracy**: 81.89%
- **Precision**: 0.97 (Pneumonia), 0.68 (Normal)
- **Recall**: 0.73 (Pneumonia), 0.97 (Normal)
- **F1-Score**: 0.83 (Pneumonia), 0.80 (Normal)

#### ✅ MobileNetV2
- **Accuracy**: 80.13%
- **Precision**: 0.98 (Normal), 0.76 (Pneumonia)
- **Recall**: 0.48 (Normal), 0.99 (Pneumonia)
- **F1-Score**: 0.64 (Normal), 0.86 (Pneumonia)

### 5.2 Sentiment Analysis

#### ✅ Feedforward Neural Network
- **Accuracy**: 84.12%

#### ✅ LSTM Network
- **Accuracy**: 83.19%
- **Precision**: 0.85 (Positive), 0.81 (Negative)
- **Recall**: 0.80 (Positive), 0.86 (Negative)
- **F1-Score**: 0.83 (both)

---

## 6. Confusion Matrices

### Medical Image Classification

#### Custom CNN
- High **precision for Pneumonia** (97%)
- High **recall for Normal** (97%)

#### MobileNetV2
- High **precision for Normal** (98%)
- Strong **recall for Pneumonia** (99%)

### Sentiment Analysis

#### Feedforward NN
- Balanced confusion matrix
- Robust for both classes

#### LSTM
- Better contextual understanding
- Slight inaccuracies in long reviews

---

## 7. Discussion & Insights

### Medical Image Classification

- **Custom CNN**: Tailored and effective but resource-intensive; prone to overfitting without regularization.
- **MobileNetV2**: Faster convergence using pretrained weights; computationally heavier during fine-tuning.

### Sentiment Analysis

- **Feedforward NN**: Simpler, fast, but lacks sequence awareness.
- **LSTM**: Captures context well, ideal for sequential data; shows room for optimization in long texts.

---

## 8. Conclusion

- Neural networks are adaptable across domains like image and text classification.
- **Transfer learning** and **LSTM architectures** outperform simpler models in complex tasks.
- Proper validation and preprocessing significantly impact performance.
- Future work: Explore **multimodal learning** combining text and image inputs.

---

## 💡 Future Directions

- Explore attention mechanisms (e.g., Transformers)
- Integrate explainability (Grad-CAM for images, SHAP/LIME for text)
- Deploy models using Flask or Streamlit for live inference

---
