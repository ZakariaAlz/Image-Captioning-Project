# **📸 Image Captioning Using Deep Learning**

## 🌟 **Overview**
This project integrates three deliverables to build a comprehensive pipeline for image analysis and caption generation:
1. **Binary Classification**: 🐾 Distinguish photos from other image types.
2. **Image Denoising**: 🖌️ Enhance image quality using convolutional autoencoders.
3. **Image Captioning**: 📝 Generate descriptive captions for images using a CNN-RNN pipeline.


✨ Features
1. 📋 Binary Classification: Identifies photos from other types of images (paintings, sketches, etc.).
2. ✨ Image Denoising: Improves the quality of noisy images using convolutional autoencoders.
3. 🖼️ Image Captioning:
   - Generates captions for images using a CNN-RNN pipeline.
   - Includes an attention mechanism for enhanced performance.
---

## **Deliverables**

### **Deliverable 1: Binary Classification**
- **Objective**: Identify photos from other types of images (paintings, sketches, etc.).
- **Methodology**:
  - Preprocess and normalize the dataset.
  - Build a Convolutional Neural Network (CNN) for classification.
  - Train and evaluate the model using training and validation data.
- **Results**:
  - Validation Accuracy: **72%**
  - Sample Predictions: Correctly identifies photos among other image types.

---

### **Deliverable 2: Image Denoising**
- **Objective**: Remove noise from images using an autoencoder.
- **Methodology**:
  - Add Gaussian noise to the dataset.
  - Train a convolutional autoencoder to reconstruct clean images from noisy inputs.
- **Results**:
  - High-quality denoised images demonstrated through visual comparisons.

---

### **Deliverable 3: Image Captioning**
- **Objective**: Generate captions for images using a deep learning model.
- **Methodology**:
  - Use a pre-trained CNN (InceptionV3) to extract image features.
  - Use a Recurrent Neural Network (RNN) to predict captions from the image features.
  - Train the model and evaluate its performance using BLEU scores.
- **Results**:
  - Sample Caption: **"A cat sitting on a couch."**
  - High BLEU scores indicating strong performance.

---

## **Technology Stack**
- 📜 **Languages**: Python
- 🧠 **Deep Learning Frameworks**: TensorFlow, Keras
- 📦 **Tools**: NumPy, Matplotlib, Pillow
- 🔍 **Pre-trained Models**:
  - InceptionV3 (for feature extraction)
  - GRU-based RNN (for text generation)
 
 ## 📂 **Dataset**
The Deliverable 1&2 have separated datasets, deliverable 3 uses the MS COCO Dataset 🐤, a widely adopted benchmark for image captioning tasks. It contains:
Source: MS COCO Dataset
### Description:
🖼️ Images with corresponding captions.
📝 Multiple annotations per image.
#### Additional Notes:
During the technical presentation, a secondary dataset was provided to test the robustness of the pipeline.


---

## 🚀 How to Run the Project
 ### 1. 📥 Clone the Repository: 
   git clone https://github.com/ZakariaAlz/Image-Captioning-Project.git 
   cd image-captioning-deep-learning

 ### 2. 📦 Install Dependencies
   Make sure you have Python 3.8+ installed. Then, run:
   pip install -r requirements.txt


 ### 3. 🔄 Preprocess the Dataset
   Place the MS COCO dataset in the data/ directory.
   Run the preprocessing script:
   python preprocess.py


 ### 4. 🏋️ Train the Model
   To train the captioning model:
   python preprocess.py

 ### 5. 🧪 Evaluate and Generate Captions
Evaluate the model on the test set and generate captions:
python evaluate.py

## 🖼️ Sample Results
Input Image
Generated Caption

Example: 🐈 A cat sitting on a couch.

### 🤝 Contributors
 👨‍💻 Zakaria ALIZOUAOUI: Model development and evaluation. <br> 
 🔍 Yacine BABACI: Dataset preprocessing and pipeline optimization. <br>
 📊 Ali MAHDJOUB: Integration and visualizations. <br>

##📜 License
 This project is licensed under the MIT License.
