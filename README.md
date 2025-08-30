Waste Classification Using Transfer Learning
Final Project for IBM AI Engineering Professional Certificate
Module 3: Building Deep Learning Models with TensorFlow
This project implements an automated waste classification system using transfer learning with VGG16 to distinguish between recyclable and organic waste products.
Project Overview
EcoClean currently lacks an efficient and scalable method to automate the waste sorting process. This project leverages machine learning and computer vision to automate the classification of waste products, improving efficiency and reducing contamination rates.
Aim
Develop an automated waste classification model that can accurately differentiate between recyclable and organic waste based on images using transfer learning techniques.
Course Information

Program: IBM AI Engineering Professional Certificate
Module: Module 3 - Building Deep Learning Models with TensorFlow
Platform: Coursera
Project Type: Final Project with 10 Required Tasks

Dataset

Source: Waste Classification Dataset
Categories: 2 classes (Organic = 0, Recyclable = 1)
Structure:
o-vs-r-split/
├── train/
│   ├── O/ (Organic waste images)
│   └── R/ (Recyclable waste images)
└── test/
    ├── O/ (Organic test images)
    └── R/ (Recyclable test images)


Technical Approach
Transfer Learning Strategy

Base Model: VGG16 pre-trained on ImageNet
Feature Extraction: Frozen VGG16 layers + custom classifier
Fine-Tuning: Unfreezing deeper layers for improved performance

Model Architecture

Input: 150x150x3 RGB images
Base: VGG16 (frozen/partially unfrozen)
Classifier:

Flatten layer
Dense(512) + ReLU + Dropout(0.3)
Dense(512) + ReLU + Dropout(0.3)
Dense(1) + Sigmoid (binary classification)



Implementation Tasks
The project is structured around 10 specific tasks required by the course:

Task 1: Print TensorFlow version
Task 2: Create test data generator
Task 3: Print training generator length
Task 4: Display model summary
Task 5: Compile the model
Task 6: Plot accuracy curves (extract features model)
Task 7: Plot loss curves (fine-tuned model)
Task 8: Plot accuracy curves (fine-tuned model)
Task 9: Visualize predictions (extract features model)
Task 10: Visualize predictions (fine-tuned model)

Key Features
Data Processing

Image Augmentation: Width/height shifts, horizontal flipping
Normalization: Pixel values scaled to [0,1]
Batch Processing: Batch size of 32

Training Configuration

Epochs: 10
Validation Split: 20%
Optimizer: RMSprop (learning rate: 1e-4)
Loss Function: Binary crossentropy

Callbacks

Early Stopping: Patience=4, monitoring validation loss
Model Checkpoint: Save best model based on validation loss
Learning Rate Scheduler: Exponential decay

Results
The project trains two models:

Extract Features Model: Frozen VGG16 as feature extractor
Fine-Tuned Model: Partially unfrozen VGG16 for better performance

Both models are evaluated on test data with classification reports and visual predictions.
Installation and Usage
Requirements
bashpip install -r requirements.txt
Running the Project
bashpython waste_classification.py
The script will:

Download and extract the dataset automatically
Train both models (extract features + fine-tuned)
Generate loss/accuracy curves
Evaluate models on test data
Display sample predictions

File Structure
├── waste_classification.py    # Main implementation
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── saved_models/            # Trained model files (generated)
    ├── O_R_tlearn_vgg16.keras
    └── O_R_tlearn_fine_tune_vgg16.keras
Learning Objectives Achieved
 Applied transfer learning using VGG16 model for image classification
 Prepared and preprocessed image data for machine learning tasks
 Fine-tuned a pre-trained model to improve classification accuracy
 Evaluated model performance using appropriate metrics
 Visualized model predictions on test data
Real-World Applications
This model can be applied to:

Municipal waste sorting facilities
Industrial waste management
Smart waste bins with automated sorting
Environmental monitoring systems

Course Completion
This project was completed as part of the IBM AI Engineering Professional Certificate program offered through Coursera. All 10 required tasks were implemented and tested according to the course specifications.
Author
Developed as part of the IBM AI Engineering Professional Certificate coursework - Module 3: Building Deep Learning Models with TensorFlow.
License
This project is for educational purposes as part of the IBM AI Engineering Professional Certificate coursework.
