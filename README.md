# Image_Classification_Project
# 🧠 AI Image Classification using CNN

A Deep Learning based Image Classification System that classifies uploaded images as **Cat 🐱** or **Dog 🐶** using a Convolutional Neural Network (CNN).

---

## 📌 Project Overview

This project uses TensorFlow and Keras to train a CNN model on a Cat vs Dog image dataset. The trained model is integrated with a Flask web application that allows users to upload images and receive real-time predictions with confidence scores.

---

## 🎯 Objectives

* Build a CNN model for image classification.
* Classify images into Cat and Dog categories.
* Visualize model performance using graphs.
* Deploy the model using Flask.
* Provide a user-friendly dashboard for predictions.

---

## 🚀 Features

* Cat vs Dog Image Classification
* CNN Deep Learning Model
* Real-Time Prediction
* Flask Web Application
* Accuracy & Loss Visualization
* Dataset Distribution Analysis
* Confusion Matrix
* Classification Report
* Responsive Dashboard UI

---

## 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* Flask
* NumPy
* Pillow
* Scikit-Learn
* Matplotlib
* Seaborn
* Werkzeug

---

## 📂 Project Structure

Image_Classification_Project/

├── app.py

├── train.py

├── requirements.txt

├── model/

│ └── cnn_model.h5

├── dataset/

│ └── animals/

│ ├── cat/

│ └── dog/

├── static/

│ ├── css/

│ ├── images/

│ ├── uploads/

│ └── graphs/

├── templates/

│ ├── home.html

│ └── index.html

└── README.md

---

## 📊 Dataset Description

Dataset contains images of cats and dogs.

* Total Images: 1000+
* Cat Images: 500
* Dog Images: 500
* Classes: 2

Folder Structure:

dataset/animals/

├── cat/

└── dog/

Image Size Used: 128 × 128

Validation Split: 20%

---

## 🔄 Data Preprocessing

The following preprocessing steps were performed:

* Image Rescaling (1/255)
* Image Resizing (128×128)
* Dataset Splitting
* Batch Generation using ImageDataGenerator

---

## 🧠 CNN Architecture

* Conv2D (32 Filters)
* MaxPooling2D
* Conv2D (64 Filters)
* MaxPooling2D
* Conv2D (128 Filters)
* MaxPooling2D
* Flatten Layer
* Dense Layer (128 Units)
* Dropout (0.5)
* Output Layer (Sigmoid)

Loss Function:
Binary Crossentropy

Optimizer:
Adam

---

## ⚙ Installation

Clone Repository:

git clone https://github.com/gayatri132004/Image_Classification_Project.git

cd Image_Classification_Project

Install Dependencies:

pip install -r requirements.txt

---

## ▶ Run Project

Train Model:

python train.py

Run Flask Application:

python app.py

Open Browser:

http://127.0.0.1:5000

---

## 📈 Results

Model generates:

* Accuracy Graph
* Loss Graph
* Dataset Distribution Graph
* Confusion Matrix
* Classification Report

Achieved Validation Accuracy:

95%+

---


## 🔮 Future Improvements

* Multi-Class Classification
* Transfer Learning (ResNet50, VGG16, EfficientNet)
* Larger Dataset
* Mobile App Integration
* Cloud Deployment
* User Authentication

---

## 📚 Learning Outcomes

Through this project, I learned:

* Deep Learning Fundamentals
* CNN Architecture
* Image Processing
* Model Evaluation
* Flask Deployment
* Dashboard Development
* GitHub Project Management

---

## 👩‍💻 Author

Gayatri Chandgude

GitHub:
https://github.com/gayatri132004
