# Employee Burnout Prediction using PyTorch

## Overview

Employee burnout is a growing problem in modern organizations. High workload, long working hours, poor job satisfaction, and continuous stress can gradually reduce employee productivity and increase attrition. Detecting burnout early allows organizations to take preventive actions before it affects both employees and business performance.

This project focuses on predicting whether an employee is likely to experience burnout using a Neural Network built with PyTorch. Instead of stopping after model training, I wanted to understand the complete machine learning lifecycle, so I also built a REST API using FastAPI and deployed the entire application on Microsoft Azure App Service.

The objective of this project was not only to build a prediction model, but also to learn how machine learning models are served in production environments.

---

# Problem Statement

Organizations collect employee-related information such as:

- Age
- Experience
- Weekly Working Hours
- Remote Work Ratio
- Satisfaction Level
- Stress Level
- Job Role

Using these features, the goal is to predict whether an employee is likely to experience burnout.

This is a **Binary Classification** problem where:

- **0 → No Burnout**
- **1 → Burnout**

---

# Project Goals

The purpose of this project was to understand the complete deployment pipeline of a machine learning application rather than only achieving high accuracy.

The project covers:

- Data preprocessing
- Feature engineering
- Neural Network implementation using PyTorch
- Model training
- Model evaluation
- Model serialization
- REST API development using FastAPI
- Cloud deployment on Microsoft Azure

---

# Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Machine Learning Utilities | Scikit-Learn |
| API Framework | FastAPI |
| API Documentation | Swagger UI |
| Cloud Platform | Microsoft Azure App Service |
| Version Control | Git |
| Repository Hosting | GitHub |

---

# Dataset Features

The model was trained using the following features:

- Age
- Gender
- Experience
- WorkHoursPerWeek
- RemoteRatio
- SatisfactionLevel
- StressLevel
- JobRole

Categorical features were converted into numerical representations before training.

- Gender → Label Encoding
- JobRole → One-Hot Encoding

---

# Model Architecture

The prediction model is a fully connected feed-forward Neural Network implemented using PyTorch.

Architecture:

Input Layer (12 Features)

↓

Hidden Layer

↓

ReLU Activation

↓

Hidden Layer

↓

ReLU Activation

↓

Output Layer

↓

Sigmoid Activation

Since the task is binary classification, the final output is a probability between **0 and 1**.

Prediction Rule:

- Probability ≥ 0.5 → Burnout
- Probability < 0.5 → No Burnout

---

# Training Pipeline

The training workflow followed these stages:

1. Data preprocessing
2. Feature encoding
3. Train/Test split
4. Tensor conversion
5. Neural Network creation
6. Forward propagation
7. Loss computation
8. Backpropagation
9. Weight optimization using Adam
10. Model evaluation
11. Save trained model

The trained model was saved using PyTorch and later reused during deployment without retraining.

---

# API Development

Instead of keeping the model inside a notebook, I built a production-style REST API using FastAPI.

Available endpoints:

### GET /

Returns API status.

### POST /predict

Accepts employee information as JSON and returns:

- Prediction
- Probability
- Confidence
- Model Version
- Timestamp

Example request:

```json
{
    "Age": 30,
    "Gender": "Male",
    "Experience": 5,
    "WorkHoursPerWeek": 45,
    "RemoteRatio": 80,
    "SatisfactionLevel": 7,
    "StressLevel": 8,
    "JobRole": "Engineer"
}
```

Example response:

```json
{
    "success": true,
    "prediction": "No Burnout",
    "probability": 0.1372,
    "confidence": 86.28,
    "model_version": "1.0.0",
    "timestamp": "2026-07-21T18:12:41Z"
}
```

---

# Cloud Deployment

The trained model was deployed using:

- FastAPI
- GitHub Actions
- Microsoft Azure App Service

Deployment pipeline:

```
Local Development

↓

Git

↓

GitHub

↓

GitHub Actions

↓

Azure App Service

↓

Live REST API
```

The application automatically redeploys whenever new commits are pushed to the main branch.

---

# Project Structure

```
employee-burnout-api/

├── app.py
├── model.py
├── preprocess.py
├── schemas.py
├── employee_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

---

# What I Learned

This project completely changed how I think about machine learning.

Initially, I believed building a model was the final objective. While working on this project, I realized that training the model is only one part of the entire engineering process.

I learned how data preprocessing directly affects model performance, how neural networks learn through forward propagation and backpropagation, how optimizers update weights, and why deployment is equally important as training.

Deploying the model to Microsoft Azure helped me understand how machine learning models are actually exposed as web services that applications can consume in real time.

More importantly, I learned that writing clean, modular code makes deployment and maintenance much easier than keeping everything inside a single notebook.

---

# Future Improvements

Some improvements I would like to explore in future versions include:

- Hyperparameter tuning
- Better feature engineering
- Model monitoring
- Docker-based deployment
- CI/CD optimization
- Containerized deployment using Azure Container Apps
- Model versioning
- Input validation improvements

---

# Repository

This project is part of my deep learning learning journey where I am building every project from scratch to understand not only how models are trained but also how they are deployed and maintained in production environments.
dm me here -https://www.linkedin.com/in/gausul-wara-643783256/