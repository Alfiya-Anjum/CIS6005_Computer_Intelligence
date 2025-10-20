# Deep Learning Applications: Medical Image Classification & Sentiment Analysis

## 1. Introduction

### 1.1 Overview
This project explores two distinct yet fundamental applications of deep learning:
- **Medical Image Classification**: Using neural networks to classify chest X-ray images into "Pneumonia" or "Normal" categories.
- **Sentiment Analysis**: Using natural language processing to classify IMDb reviews as positive or negative.

The study focuses on evaluating different neural network architectures and their performance on both tasks, highlighting their strengths and challenges in real-world applications.

## 2. Dataset Preparation

### 2.1 Medical Image Classification Dataset

**Dataset Source**: Chest X-ray Images (Pneumonia) dataset from Kaggle, contributed by Paul Mooney  
**URL**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

- **Total images**: 5,863
- **Classes**: Pneumonia and Normal
- **Subdivisions**:
  - Training set: 5,216 images
  - Validation set: 16 images
  - Testing set: 624 images

**Preprocessing Steps**:
1. **Image Resizing**:
   - All images were resized to 150x150 pixels for uniformity and compatibility with the input dimensions of the models.
2. **Grayscale Conversion**:
   - Images were converted to grayscale to reduce complexity and focus on critical features.
3. **Normalization**:
   - Pixel values were scaled to the range [0, 1] for better numerical stability during training.
4. **Data Augmentation**:
   - To enhance model generalization, the training data underwent transformations such as:
     - Rotation: Up to 30 degrees.
     - Zooming: Random zoom up to 20%.
     - Shifting: Horizontal and vertical shifts up to 10%.
     - Flipping: Horizontal flips.

**Loading and Preprocessing Data:**

<img width="975" height="346" alt="image" src="https://github.com/user-attachments/assets/58d262b8-03ec-4bec-a5c5-6bd263cb7e62" />
<img width="975" height="498" alt="image" src="https://github.com/user-attachments/assets/b0c2320f-d191-4fd9-b788-39bd7096b839" />

**Data Augmentation:**

<img width="975" height="206" alt="image" src="https://github.com/user-attachments/assets/acdd61f9-8bdb-4ae6-981e-96b6e6c5c13f" />

<img width="745" height="872" alt="image" src="https://github.com/user-attachments/assets/7400c6e7-41cb-4b2b-a79a-3fa13b9af452" />

*Sample preprocessed chest X-ray images showing Pneumonia and Normal categories, resized to 150x150 pixels and converted to grayscale.*

### 2.2 Sentiment Analysis Dataset

**Dataset Source**: IMDb movie reviews dataset.

**Dataset Description**:
- **Total reviews**: 50,000
- **Classes**: Positive and Negative
- **Split**:
  - Training set: 25,000 reviews
  - Testing set: 25,000 reviews

**Preprocessing Steps**:
1. **Tokenization**:
   - Each review was tokenized into sequences of integers representing words.
2. **Padding**:
   - Sequences were padded to a uniform length of 200 words.
3. **Vocabulary Size**:
   - Limited to the top 10,000 most frequent words in the dataset.

**Code:**

<img width="975" height="372" alt="image" src="https://github.com/user-attachments/assets/aa896a9f-fe78-4bae-a081-2eb1ddbba12a" />


## 3. Model Creation and Training

### 3.1 Medical Image Classification

#### Model 1: Custom CNN

**Architecture**:
- **Convolutional Layers**:
  - The model includes multiple convolutional layers with 32, 64, 128, and 256 filters, each using ReLU activation for non-linear transformation.
- **Regularization Techniques**:
  - Batch normalization layers are used to stabilize training and improve convergence speed.
  - Dropout layers are applied to reduce overfitting, especially in dense layers.
- **Fully Connected Layers**:
  - The flattened output of the convolutional layers is passed through dense layers.
  - The final dense layer uses a sigmoid activation function for binary classification.

<img width="975" height="488" alt="image" src="https://github.com/user-attachments/assets/a4ab89da-f950-460e-9d9a-94742aca8651" />

<img width="975" height="523" alt="image" src="https://github.com/user-attachments/assets/a42c49e8-97e9-42d4-8d56-4cbf6e6be5df" />

<img width="765" height="645" alt="image" src="https://github.com/user-attachments/assets/f6866dd6-7cc4-4bc5-9dd3-7ab2300b80de" />

*The following figure demonstrates the training and validation performance of the Custom CNN model over 12 epochs*


#### Model 2: Transfer Learning with MobileNetV2

**Architecture**:
- **Base Model**: Pretrained MobileNetV2 is used as a feature extractor. The initial layers are frozen to retain learned features from ImageNet.
- **Custom Layers**:
  - A global average pooling layer reduces the spatial dimensions of the feature maps.
  - Dense layers are added to adapt the model to binary classification for pneumonia detection.
  - Dropout layers help prevent overfitting.
- **Fine-tuning**: Certain layers of MobileNetV2 are unfrozen to refine weights for the specific dataset.

**Code Snippit:**

<img width="975" height="242" alt="image" src="https://github.com/user-attachments/assets/864fae0b-a480-4fc3-b8ee-8bfff7dea363" />

<img width="975" height="802" alt="image" src="https://github.com/user-attachments/assets/475a86cc-8796-4683-b95a-1a6986b7af5a" />

<img width="761" height="647" alt="image" src="https://github.com/user-attachments/assets/d962a300-5bac-4105-b543-77e86bbe7dab" />

<img width="797" height="897" alt="image" src="https://github.com/user-attachments/assets/278f644e-905b-4490-8eff-772aa6e99b45" />

*The training and fine-tuning process for MobileNetV2 is visualized in the following figures, showing accuracy and loss trends over epochs*

### 3.2 Sentiment Analysis

#### Model 1: Feedforward Neural Network

**Architecture**:
- **Embedding Layer**: Converts words into dense vector representations of size 128.
- **Flatten Layer**: Flattens the embedding vectors for input to the dense layers.
- **Dense Layers**:
  - Two dense layers with ReLU activation process the features.
  - The final dense layer uses a sigmoid activation function for binary classification.

<img width="975" height="250" alt="image" src="https://github.com/user-attachments/assets/61dbe5ce-8e2f-458a-a92e-e6bdf73739f5" />

<img width="752" height="727" alt="image" src="https://github.com/user-attachments/assets/f3619ec2-d103-4033-b8f5-8f7d026c55d4" />

**Figure illustrating the training and validation accuracy of the Feedforward Neural Network for IMDb sentiment analysis.**

#### Model 2: LSTM-based Network

**Architecture**:
- **Embedding Layer**: Converts words into dense vector representations of size 64.
- **LSTM Layers**:
  - A bidirectional LSTM with 64 units captures sequential dependencies in the text.
  - A second LSTM layer with 32 units refines the temporal understanding.
- **Dropout Layers**:
  - Used between layers to prevent overfitting.
- **Dense Layer**:
  - A single neuron with sigmoid activation classifies the sentiment.

**Code Snippet:**

<img width="975" height="433" alt="image" src="https://github.com/user-attachments/assets/78d75a7f-0359-42db-93f9-41f19e5c76d6" />

<img width="738" height="902" alt="image" src="https://github.com/user-attachments/assets/90863810-77a0-418d-bbe8-1061003aff3c" />

*The figure highlights the training and validation accuracy achieved by the LSTM model for IMDb sentiment analysis*

## 4. Hyperparameters and Training Process

### 4.1 Hyperparameters

#### Medical Image Classification

| Model | Optimizer | Learning Rate | Batch Size | Epochs |
|-------|-----------|---------------|------------|--------|
| Custom CNN | Adam | 0.0003 | 32 | 12 |
| MobileNetV2 | Adam | 0.0001 | 32 | 10 + 15 (fine-tuning) |

**Custom CNN**
<img width="975" height="721" alt="image" src="https://github.com/user-attachments/assets/311bfe17-bd2e-41e9-a9b5-48f6f8bbbdb3" />

**Transfer Learning with MobileNetV2:**
<img width="975" height="417" alt="image" src="https://github.com/user-attachments/assets/984a6cc6-3882-4b26-803a-e44c70815eea" />


#### Sentiment Analysis

| Model | Optimizer | Batch Size | Epochs |
|-------|-----------|------------|--------|
| Feedforward NN | Adam | 32 | 5 |
| LSTM | Adam | 32 | 5 |

**Feedforward Neural Network and LSTM**

<img width="975" height="233" alt="image" src="https://github.com/user-attachments/assets/84b0b666-3368-475c-8da7-f0960f897918" />


### 4.2 Training Process

#### Medical Image Classification
1. Data is split into training, validation, and test sets.
2. For Custom CNN:
   - Data augmentation using ImageDataGenerator applied to the training set.
   - Training is done with callbacks like ReduceLROnPlateau to dynamically adjust learning rate based on validation accuracy.
3. For MobileNetV2:
   - Pretrained MobileNetV2 used for feature extraction.
   - Fine-tuned by unfreezing layers starting from the 100th layer.

#### Sentiment Analysis
1. Text reviews are tokenized and padded to ensure uniform input dimensions.
2. Models are trained with a 20% validation split from the training data.
3. Performance metrics are visualized to monitor training and validation accuracy and loss.

## 5. Model Evaluation and Results

### 5.1 Medical Image Classification

#### Custom CNN:
- **Test Accuracy**: 81.89%
- **Precision**: 0.97 (Pneumonia), 0.68 (Normal)
- **Recall**: 0.73 (Pneumonia), 0.97 (Normal)
- **F1-Score**: 0.83 (Pneumonia), 0.80 (Normal)


- <img width="975" height="274" alt="image" src="https://github.com/user-attachments/assets/77e56268-2d5b-4f57-b253-6465e51b8823" />


#### MobileNetV2:
- **Test Accuracy**: 80.13%
- **Precision**: 0.98 (Normal), 0.76 (Pneumonia)
- **Recall**: 0.48 (Normal), 0.99 (Pneumonia)
- **F1-Score**: 0.64 (Normal), 0.86 (Pneumonia)


- <img width="927" height="272" alt="image" src="https://github.com/user-attachments/assets/d73e3dc1-70ad-47b0-8723-3b168e5dd57a" />


### 5.2 Sentiment Analysis

#### Feedforward Neural Network:
- **Test Accuracy**: 84.12%


- <img width="950" height="275" alt="image" src="https://github.com/user-attachments/assets/301b9fa4-67ff-4af0-8004-88a7cd98ffe4" />


#### LSTM-based Network:
- **Test Accuracy**: 83.19%
- **Precision**: 0.85 (Positive), 0.81 (Negative)
- **Recall**: 0.80 (Positive), 0.86 (Negative)
- **F1-Score**: 0.83 (Positive), 0.83 (Negative)


- <img width="975" height="195" alt="image" src="https://github.com/user-attachments/assets/f34bf8c8-bf6c-43af-a5b4-3842b7988eb1" />


## 6. Confusion Matrices

### 6.1 Medical Image Classification

#### Confusion Matrix for Custom CNN Model:
- **Description**: This confusion matrix evaluates the performance of the custom CNN model on the test dataset. It shows the number of true positives (correctly predicted Pneumonia), true negatives (correctly predicted Normal), false positives (Normal misclassified as Pneumonia), and false negatives (Pneumonia misclassified as Normal).
- **Insights**:
  - Precision for Pneumonia: 97% (high precision indicates the model minimizes false positives for Pneumonia)
  - Recall for Normal: 97% (the model effectively identifies Normal cases but struggles slightly with Pneumonia recall)

<img width="975" height="730" alt="image" src="https://github.com/user-attachments/assets/4e577ae0-04d5-4735-ab00-317648af9271" />


#### Confusion Matrix for MobileNetV2 Model:
- **Description**: This confusion matrix evaluates the transfer learning-based MobileNetV2 model's performance. MobileNetV2 provides slightly better generalization but struggles with Normal class recall.
- **Insights**:
  - Normal class precision: 98% (high confidence when predicting Normal cases)
  - Pneumonia class recall: 99% (fewer false negatives, ensuring Pneumonia detection)

<img width="975" height="974" alt="image" src="https://github.com/user-attachments/assets/a4a09364-c24a-41e5-84e7-438bc546063d" />


### 6.2 Sentiment Analysis

#### Confusion Matrix for Feedforward Neural Network:
- **Description**: This matrix illustrates the model's performance on binary classification for sentiment analysis (positive or negative reviews).
- **Insights**:
  - Accuracy: 98%
  - False positives and false negatives are balanced, indicating the model's robustness in identifying sentiments

<img width="967" height="1231" alt="image" src="https://github.com/user-attachments/assets/893b1c47-25a2-4dfd-b866-d3323dc67151" />


#### Confusion Matrix for LSTM Model:
- **Description**: The confusion matrix highlights the LSTM model's performance, with better recall and precision due to its ability to understand sequential dependencies.
- **Insights**:
  - Precision for Negative: 81%
  - Recall for Positive: 80%
  - The LSTM captures context well, but occasional errors are observed in longer reviews

<img width="856" height="1103" alt="image" src="https://github.com/user-attachments/assets/6a69451b-6c54-4dd1-85da-e7308499b46b" />


## 7. Discussion and Insights

### 7.1 Medical Image Classification

#### Custom CNN:
I found that the Custom CNN performed well for the dataset, especially given its tailored architecture. It was particularly effective with limited data and managed to achieve decent accuracy. However, I noticed that it required a lot of effort to train and needed regularization to avoid overfitting. This made the training process more time-consuming and challenging for me.

#### MobileNetV2:
MobileNetV2 was impressive because it leveraged its pretrained knowledge from ImageNet, which sped up the convergence. I liked how it achieved good results with less training compared to the Custom CNN. That said, fine-tuning the model was computationally expensive, and I found it a bit tricky to manage with my system's resources.

### 7.2 Sentiment Analysis

#### Feedforward Neural Network:
The Feedforward Neural Network was much simpler to understand and implement, which I appreciated as it made the process less overwhelming. However, its simplicity came at a cost. I noticed it struggled to capture the sequential nature of text data, and its performance was limited as a result.

#### LSTM-based Network:
The LSTM-based model stood out for its ability to understand the context and long-term dependencies in the text, which made it better suited for sentiment analysis. I was especially impressed by how well it handled complex relationships in the reviews. However, I did notice occasional errors in longer reviews, which might indicate that there's room for improvement in tuning the model further.

## 8. Conclusion

- Neural networks demonstrate versatility across diverse tasks like image classification and text analysis.
- Transfer learning and LSTM-based architectures outperform simpler models in respective tasks.
- Validation sets are crucial for monitoring overfitting and fine-tuning.
- Future work could explore multimodal learning combining text and image data.

## 💡 Future Directions

- Explore attention mechanisms (e.g., Transformers)
- Integrate explainability (Grad-CAM for images, SHAP/LIME for text)
- Deploy models using Flask or Streamlit for live inference
