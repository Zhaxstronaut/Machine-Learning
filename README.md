# **AI-Based Network Traffic Classification** 🚀

## **Overview** 🌐
This project focuses on developing and evaluating machine learning models to classify network traffic based on features extracted from datasets. It leverages multiple classification algorithms to predict traffic types, providing insights into model effectiveness. The final models are saved for future predictions and can be easily integrated into real-world applications.

---

## **Table of Contents** 📚
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Model Prediction](#model-prediction)
- [Visualization and Analysis](#visualization-and-analysis)
- [Model Saving and Deployment](#model-saving-and-deployment)
- [Features](#features)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

---

## **Introduction** 📝
Network traffic classification is essential in cybersecurity, network management, and Quality of Service (QoS) provisioning. This project utilizes advanced machine learning techniques to classify network traffic into categories. The primary objective is to compare the performance of several models to determine the most effective approach.

---

## **Project Structure** 🏗️
- **Data Loading and Preprocessing:** Scripts for loading, cleaning, and preprocessing network datasets.
- **Model Development:** Implementations of Random Forest, Gaussian Naive Bayes, Logistic Regression, K-Nearest Neighbors (KNN), and Multi-Layer Perceptron (MLP).
- **Evaluation:** Model evaluation using accuracy, confusion matrices, ROC curves, and learning curves.
- **Visualization:** Graphical representations of performance metrics.
- **User Input and Prediction:** Interface for predicting traffic type based on user input.
- **Model Saving:** Trained models are saved for deployment.

---

## **Datasets** 📊
The project uses multiple datasets sourced from network traffic, including:
- **Ping Dataset**
- **Voice Dataset**
- **DNS Dataset**
- **Telnet Dataset**

These datasets are loaded from Google Drive, concatenated, and preprocessed to form a unified dataset for training and evaluation.

---

## **Installation** ⚙️

### **1. Clone the Repository:**

```bash
# Clone This Repo
https://github.com/Zhaxstronaut/Machine-Learning.git

# Go to Directory
cd Machine-Learning
```
### 2. **Install Dependencies:**

Ensure that you have Python 3.x installed. Then, install the required packages using pip:

```python 
pip install -r requirements.txt
```
 ### 3. **Dataset Download:**

The datasets are automatically downloaded and loaded into the project from Google Drive using the provided URLs in the code. No manual intervention is required.

## Usage 🛠️

### Data Preprocessing 🧹

Data preprocessing is a crucial step that includes cleaning the datasets, handling missing values, and standardizing the features. The following steps are carried out:

- **Concatenation:** The datasets are merged into a single dataframe.
- **Feature Engineering:** Irrelevant features such as 'Forward Packets', 'Forward Bytes', 'Reverse Packets', and 'Reverse Bytes' are dropped.
- **Label Encoding:** The target variable 'Traffic Type' is encoded into categorical codes.

### Model Training and Evaluation 🎯
The project trains and evaluates several machine
learning models:

1. **Model Initialization:**

- Random Forest Classifier
- Gaussian Naive Bayes
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)

2. **Training:**

- The dataset is split into training and testing sets.
- Each model is trained on the training set.

3. **Evaluation:**

- Models are evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC AUC.
- Confusion matrices and ROC curves are generated for a detailed analysis of model performance.
- Learning curves are plotted to observe the model's performance as the training data size increases.

### Model Prediction 🔍
The project includes a user interface to input new data for traffic classification. The following steps are followed:

- **User Input:** Users are prompted to input values for specific features.
- **Preprocessing:** The user input is preprocessed using the same scaler applied during model training.
- **Prediction:** Each model provides a prediction for the traffic type based on the input data.

## Visualization and Analysis 📈
The project provides various visual tools to understand model performance:

- **Confusion Matrix:** Visual representation of the true vs. predicted classifications.
- **ROC Curve:** Displays the trade-off between the true positive rate and false positive rate for each class.
- **Learning Curve:** Shows the performance of the model on the training and validation sets as the amount of training data varies.

## Model Saving and Deployment 💾
The trained models are saved using Python’s pickle module, allowing for easy deployment in production environments. This feature is particularly useful for integrating the classification models into larger systems or applications where network traffic classification is required

## Features ✨
- Comprehensive Preprocessing: Includes handling of missing data, feature selection, and scaling.
- Multiple Classifiers: Implements and compares five different machine learning models.
- Detailed Evaluation: Provides a thorough evaluation using multiple metrics and visualizations.
- User Interaction: Allows for dynamic predictions based on user input.
- Model Persistence: Saves models for future use, making deployment straightforward.

## Results 🏆
The project outputs detailed results for each model, including:

- Accuracy: Overall accuracy of each model on the test dataset.
- Classification Reports: Detailed performance metrics for each traffic type.
- Confusion Matrices: Provides insights into model misclassifications.
- ROC Curves: Evaluates the model’s ability to distinguish between different traffic types.
- Learning Curves: Shows how each model’s performance scales with the amount of training data.

## Conclusion 🎉
This project successfully demonstrates the application of machine learning to network traffic classification. By comparing multiple models, we gain insights into the strengths and weaknesses of different approaches. The saved models are ready for deployment and can be used in real-time traffic classification systems.

## License 📜
This project is licensed under the MIT License. You are free to use, modify, and distribute the code as per the terms of the license. See the LICENSE file for details.
