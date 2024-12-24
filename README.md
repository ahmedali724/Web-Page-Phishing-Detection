# Web-Page-Phishing-Detection

This project focuses on building a machine learning-based system to detect phishing websites. It leverages various traditional machine learning models and deep learning architectures to classify URLs as either 'legitimate' or 'phishing' based on their features.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Implementation](#implementation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Acknowledgments](#acknowledgments)

---

## Introduction
Phishing attacks are a significant cybersecurity threat where attackers deceive users into providing sensitive information. This project aims to address this challenge by:
- Developing robust machine learning models for phishing detection.
- Comparing multiple algorithms to determine the most effective approach.
- Evaluating models using comprehensive metrics.
- Laying the groundwork for potential real-world deployment.

---

## Dataset
The dataset includes 11,430 records with 89 features, grouped as:
- **URL Composition Features**: Metrics describing the structure and syntax of the URL.
- **Domain and Host Features**: Details about the domain's age, behavior, and reputation.
- **Content and Behavior Features**: Visual elements, user interactions, and external references.

The target variable is the classification of URLs as "phishing" or "legitimate."

---

## Methodology
Key steps in the methodology include:
1. **Feature Selection**: Using Particle Swarm Optimization to select relevant features.
2. **Model Selection**: Testing multiple models:
   - Logistic Regression
   - Support Vector Machines (SVM)
   - K-Nearest Neighbors (KNN)
   - Decision Trees
   - Random Forest
   - Neural Networks (CNN, RNN)
3. **Training and Testing**:
   - Data split into 80% training and 20% testing sets.
   - K-fold cross-validation (k=10) for reliable evaluation.
   - Hyperparameter optimization using Grid Search.

---

## Implementation
### Key Libraries:
- **Data Handling**: `pandas`, `numpy`
- **Modeling**: `sklearn`, `TensorFlow/Keras`
- **Evaluation**: `sklearn.metrics`

### Steps:
1. Preprocess data (scaling, encoding, and splitting).
2. Train multiple models with selected features.
3. Evaluate models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

---

## Evaluation Metrics
Models were assessed using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Curve**
- **Training and Testing Times**

---

## Results
The top-performing models:
- **Random Forest**: Highest accuracy (95.06%) and AUC (99.06%).
- **Logistic Regression**: Fastest training time, making it ideal for real-time applications.
- **CNN**: Balanced performance with strong feature extraction capabilities.

---

## Technologies Used
- Python
- scikit-learn
- TensorFlow/Keras
- Pandas and NumPy

---

## Acknowledgments
This project was completed as part of the **Network Security Course**, under the guidance of the Faculty of Engineering, Helwan University. Special thanks to the instructors and team members for their support and collaboration.

