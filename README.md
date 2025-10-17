# 🧠 Brain Tumor Segmentation and Classification using MRI & CT Images

This project implements a **deep learning-based pipeline** for brain tumor detection, segmentation, and classification using **MRI and CT scan images**.  
It leverages **EfficientNetB3** and **ResNet50** architectures for feature extraction and classification, and integrates **Grad-CAM** for visual explainability.  
The workflow is designed to run seamlessly in **Google Colab** using **Python, OpenCV, SimpleITK, and PyTorch/Keras** frameworks.

---

## 📘 Overview

Brain tumors can be life-threatening if not detected early. Manual diagnosis using MRI/CT scans is time-consuming and prone to human error.  
This project automates the process by:

- Preprocessing and normalizing medical images  
- Segmenting the tumor region using **nnU-Net / Deep learning models**  
- Extracting deep features using **EfficientNetB3 (MRI)** and **ResNet50 (CT)**  
- Classifying tumor presence and type  
- Visualizing model attention with **Grad-CAM**

---

## 🧩 Dataset

### MRI Dataset
- **Source:** BraTS Challenge Dataset (Kaggle)
- **Modality:** MRI (T1, T1ce, T2, FLAIR)
- **Samples:** 369 training subjects, 125 validation subjects
- **Classes:** Enhancing Tumor, Tumor Core, Whole Tumor, Background

### CT Dataset
- **Source:** TCIA / REMBRANDT collections (Kaggle)
- **Modality:** CT scans with patient metadata
- **Classes:** Tumor vs. Non-tumor
- **Purpose:** Binary classification of tumor presence

---

## ⚙️ Methodology

### 1. **Preprocessing**
- Skull stripping and intensity normalization using **OpenCV** and **SimpleITK**
- Image resizing and channel standardization
- Data augmentation (rotation, flipping, scaling)
- Splitting dataset into training, validation, and test sets

### 2. **Segmentation**
- Applied **2D nnU-Net** for CT and **3D nnU-Net** for MRI
- Outputs tumor masks for each scan
- Evaluated using Dice and IoU metrics

### 3. **Feature Extraction & Classification**
| Modality | Model | Purpose | Framework |
|-----------|--------|----------|------------|
| MRI | EfficientNetB3 | Feature extraction & classification | TensorFlow / Keras |
| CT | ResNet50 | Feature extraction & classification | TensorFlow / Keras |

- Optimizer: Adam  
- Loss Function: Binary / Categorical Crossentropy  
- Metrics: Accuracy, Precision, Recall, F1-score, AUC

### 4. **Explainability (Grad-CAM)**
- Visualized activated regions in MRI & CT images  
- Highlighted tumor-focused areas in correct predictions  

---

## 🧪 Experimental Setup

| Parameter | Value |
|------------|--------|
| Learning Rate | 1e-4 |
| Batch Size | 16 |
| Epochs | 50 |
| Optimizer | Adam |
| Loss | CrossEntropy |
| Validation Split | 20% |

**Training Environment:**  
- Google Colab with GPU runtime  
- Python 3.10  
- TensorFlow / PyTorch  
- Keras, OpenCV, SimpleITK, NumPy, Matplotlib  

---

## 📈 Results

### MRI Classification (EfficientNetB3)
| Metric | Score |
|--------|--------|
| Accuracy | 0.985 |
| Precision | 0.972 |
| Recall | 0.978 |
| F1-Score | 0.975 |
| AUC | 0.991 |

### CT Classification (ResNet50)
| Metric | Score |
|--------|--------|
| Accuracy | 0.981 |
| Precision | 0.967 |
| Recall | 0.974 |
| F1-Score | 0.970 |
| AUC | 0.988 |

### Segmentation (nnU-Net)
| Metric | MRI (3D) | CT (2D) |
|--------|-----------|----------|
| Dice Coefficient | 0.812 | 0.845 |
| IoU | 0.765 | 0.791 |

---

## 🎯 Grad-CAM Visualization

The Grad-CAM heatmaps reveal that the model focuses on tumor-affected regions, validating model interpretability.

| MRI Example | CT Example |
|--------------|------------|
| ![MRI GradCAM](assets/mri_gradcam.png) | ![CT GradCAM](assets/ct_gradcam.png) |

---

## 🚀 How to Run (Google Colab)

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/brain-tumor-segmentation-classification.git
   cd brain-tumor-segmentation-classification
   ```
   ---
2. **Upload to Google Colab**

3. **Open the .ipynb file in Colab.**

Enable GPU Runtime: Runtime → Change runtime type → GPU.

Install dependencies

!pip install tensorflow keras opencv-python simpleitk efficientnet pytorch torchvision matplotlib numpy


4. **Run all cells**

Preprocessing → Segmentation → Classification → Grad-CAM

Visualize results and metrics

---

**📚 References**

Menze et al., “The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)”, IEEE Trans. Med. Imaging, 2015.

Bakas et al., “Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation”, Frontiers in Neuroscience, 2018.

Isensee et al., “nnU-Net: Self-Adapting Framework for U-Net-Based Segmentation”, Nature Methods, 2021.

Tan & Le, “EfficientNet: Rethinking Model Scaling for CNNs”, ICML, 2019.

---

**🧭 Future Work**

Integrate multimodal fusion (MRI + CT)

Apply 3D transformer-based models for improved segmentation

Clinical validation using real hospital datasets

---

**👩‍💻 Author**

Navya Srija
Master’s in Computer Science, Southern Illinois University Edwardsville
📧 navyasrija77@gmail.com


