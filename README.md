**🧠 Brain Tumor Detection Using Deep Learning**

**📘 Overview**

The Brain Tumor Detection System leverages a ResNet50 deep learning model to analyze MRI images and classify brain tumors into four categories — Meningioma, Glioma, Pituitary, or None.
The project is built using Flask as the backend framework, providing an interactive web interface where users can upload MRI scans and receive accurate diagnostic predictions instantly.
________________________________________
**🚀 Features**

1.	🧩 Deep Learning-Based Classification — Uses a fine-tuned ResNet50 CNN model for high-accuracy brain tumor detection.
2.	🖼️ Image Upload & Processing — Upload MRI scans directly through the web app for on-the-fly analysis.
3.	⚡ GPU-Optimized Inference — Automatically detects GPU (CUDA) availability for faster processing.
4.	🔒 Model Download Option — Allows downloading the trained model for offline analysis or retraining.
5.	🌐 Flask Web Interface — Simple, responsive UI built with HTML, CSS, and Flask templates.
________________________________________
**🧠 Model Architecture**

•	Base Model: ResNet50 (pretrained on ImageNet)
•	Modified Layers:
o	Fully connected layers replaced with custom dense layers
o	Activation: SELU
o	Regularization: Dropout (p=0.4)
o	Output: 4 neurons (for 4 tumor classes)
o	Final Activation: LogSigmoid
•	Model File: bt_resnet50_model.pt

**🧩 Project Structure**

**Brain_Tumor_Detection**/
│
├── app.py                      # Flask app (main backend)
├── models/
│   └── bt_resnet50_model.pt    # Trained ResNet50 model
├── static/
│   └── photos/                 # Uploaded MRI images
├── templates/
│   ├── DiseaseDet.html         # Home page
│   ├── uimg.html               # Upload page
│   ├── pred.html               # Prediction result
│   └── error.html              # Error handling
├── requirements.txt
└── README.md

