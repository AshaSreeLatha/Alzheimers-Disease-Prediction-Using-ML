# 🧠 Alzheimer's Disease Prediction Using Machine Learning Models

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Django](https://img.shields.io/badge/Django-4.x-green?style=for-the-badge&logo=django)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> A machine learning-powered web application that predicts Alzheimer's disease using supervised learning models trained on clinical and demographic data — built with Django and Scikit-Learn.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [ML Models Used](#ml-models-used)
- [Dataset](#dataset)
- [License](#license)

---

## 📖 About the Project

Alzheimer's disease is a progressive neurological disorder affecting millions worldwide. Early detection is crucial for better management and care. This project presents a **web-based prediction system** where users can input clinical parameters and receive a prediction about Alzheimer's disease risk using multiple trained ML models.

---

## ✨ Features

- 🔐 Role-based Access — Separate Admin and User login portals
- 📊 Dataset Management — Upload and view the Alzheimer's dataset
- 🤖 Model Training — Train multiple ML models directly from the UI
- 🔮 Disease Prediction — Predict Alzheimer's risk based on input features
- 📈 Model Comparison — Compare accuracy of different ML algorithms
- 📋 User Management — Admin can view registered users

---

## 🛠️ Tech Stack

| Layer      | Technology                               |
| ---------- | ---------------------------------------- |
| Backend    | Python, Django                           |
| Frontend   | HTML, CSS, Bootstrap, JavaScript, jQuery |
| ML Library | Scikit-Learn, Pandas, NumPy              |
| Database   | SQLite3                                  |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**

```bash
   git clone https://github.com/AshaSreeLatha/Alzheimers-Disease-Prediction-Using-ML.git
   cd Alzheimers-Disease-Prediction-Using-ML
```

2. **Install dependencies**

```bash
   pip install -r requirements.txt
```

3. **Apply migrations**

```bash
   python manage.py makemigrations
   python manage.py migrate
```

4. **Run the server**

```bash
   python manage.py runserver
```

5. **Open in browser**

```
   http://127.0.0.1:8000/
```

---

## 🤖 ML Models Used

| Model                  | Description                |
| ---------------------- | -------------------------- |
| Logistic Regression    | Baseline linear classifier |
| Decision Tree          | Rule-based tree classifier |
| Random Forest          | Ensemble of decision trees |
| Support Vector Machine | Margin-based classifier    |
| K-Nearest Neighbors    | Distance-based classifier  |

---

## 📊 Dataset

The dataset contains clinical and demographic features:

- Age, Gender
- MMSE Score
- Functional Assessment
- Memory Complaints
- Behavioral Problems
- ADL (Activities of Daily Living)
- Diagnosis (Target variable)

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Asha Sree Latha**  
[![GitHub](https://img.shields.io/badge/GitHub-AshaSreeLatha-black?style=flat&logo=github)](https://github.com/AshaSreeLatha)

---

⭐ _If you found this project helpful, please give it a star!_
