# Machine Learning Study
This repository consists of the ML projects regarding the university's Spring 2025 course.

***course links:***
 - https://catalog.bilkent.edu.tr/course/c11464.html
 - http://ciceklab.cs.bilkent.edu.tr/syllabus/2024-2025-spring/cs-464-introduction-to-machine-learning

<br>


<br>

## PCA for Cat-Dog Image Morphing
This project involves the implementation of Principal Component Analysis (PCA), focusing on dimensionality reduction and image morphing. The project processes a dataset of 4000 grayscale cat and dog images (60x60 pixels) using Singular Value Decomposition (SVD) to compute the first 10 principal components, calculate their Proportion of Variance Explained (PVE), and visualize them as eigenfaces. Additionally, it performs face morphing between a cat and a dog image by interpolating their projections in eigenspace, generating intermediate images for t values from 0 to 1 in 0.1 increments. The code, written in Python using NumPy and Matplotlib, demonstrates proficiency in linear algebra, data preprocessing, and visualization techniques.

![image](https://github.com/user-attachments/assets/7cac916b-b24a-44bf-b143-c549d7458c89)

![image](https://github.com/user-attachments/assets/6ea80309-3be0-475c-a887-5ee518b834db)

<br>


<br>

## Logistic Regression for Maternal Health Risk Classification
Implemented a Logistic Regression model using Batch Gradient Ascent, applied to the Maternal Health Risk dataset. The project classifies maternal health risk as low (0) versus moderate/high (1) using six features: Age, SystolicBP, DiastolicBP, BS, BodyTemp, and HeartRate. The model is trained for 1000 iterations with learning rates {10⁻³, 10⁻², 10⁻¹, 1, 10}, incorporating feature normalization and a bias term. Validation accuracy is plotted against iterations to select the best learning rate, and the test set performance is evaluated with accuracy and a confusion matrix. Written in Python using NumPy, Pandas, and Matplotlib, this project demonstrates proficiency in gradient-based optimization, binary classification, and performance evaluation.

![image](https://github.com/user-attachments/assets/6ff4d892-7aaf-43d1-bda0-aacbd3f4663a)



<br>


<br>

## Support Vector Machines for Maternal Health Risk Classification
Built a linear Support Vector Machine (SVM) classifier with a soft margin, using the Maternal Health Risk dataset. The project classifies maternal health risk as low (0) versus moderate/high (1) based on six features: Age, SystolicBP, DiastolicBP, BS, BodyTemp, and HeartRate. A custom 5-fold cross-validation is implemented to select the best hyperparameter C from {0.001, 0.01, 0.1, 1, 10}, evaluating accuracy across folds. The final model, trained with the optimal C, reports test accuracy, confusion matrix, precision, recall, and F1-score. Written in Python using scikit-learn for SVM training, NumPy, Pandas, and custom cross-validation logic, this project highlights expertise in model evaluation, hyperparameter tuning, and classification performance analysis.

![image](https://github.com/user-attachments/assets/a9d07644-8fbe-4a89-921f-6baab1b79064)

![image](https://github.com/user-attachments/assets/a968101b-ac91-4683-9f2d-a04fe3c50d9f)



<br>


<br>

## MLP, CNN & Transfer Learning for Hieroglyph Classification

This project involves designing and implementing three neural networks in PyTorch for classifying images of ancient Egyptian hieroglyphs from a Kaggle dataset. The models include a Multi-Layer Perceptron (MLP), a Convolutional Neural Network (CNN), and a fine-tuned ResNet18, each trained on a dataset of 128x128 images across 20 classes. The implementation covers data preprocessing, custom training/evaluation loops, hyperparameter tuning (learning rate and optimizer), and transfer learning.

Performance varied significantly across architectures: the MLP achieved a modest 45.92% accuracy due to its inability to capture spatial features. The custom CNN performed much better, reaching 81.63% accuracy by leveraging convolutional and pooling layers to learn spatial hierarchies. The fine-tuned ResNet18 model demonstrated the power of transfer learning, achieving the highest accuracy of 92.86% by adapting its pre-trained features to the specific hieroglyph dataset. The project highlights proficiency in deep learning fundamentals, model architecture design, hyperparameter optimization, and the effective application of transfer learning for image classification tasks.
