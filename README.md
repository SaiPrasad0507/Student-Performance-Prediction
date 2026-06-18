# 🎓 Student Performance Prediction Using Machine Learning

## 📌 Project Overview

Student Performance Prediction is a Machine Learning project that aims to predict students' final academic marks based on factors such as study hours, attendance percentage, sleep duration, previous academic scores, extracurricular participation, and parental education level.

The project follows a complete Machine Learning workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and prediction. Multiple regression models are implemented and compared to identify the best-performing model.

---

## 🎯 Objectives

* Predict student final marks using machine learning techniques.
* Analyze factors affecting academic performance.
* Compare the performance of multiple regression algorithms.
* Build a predictive system for educational performance analysis.

---

## 📊 Dataset Description

The dataset contains **500 student records** with the following attributes:

| Feature           | Description                                 |
| ----------------- | ------------------------------------------- |
| student_id        | Unique student identifier                   |
| gender            | Student gender                              |
| school_type       | Type of school                              |
| study_hours_daily | Daily study hours                           |
| attendance_pct    | Attendance percentage                       |
| sleep_hours       | Average sleep hours                         |
| previous_score    | Previous academic score                     |
| extracurricular   | Participation in extracurricular activities |
| parent_education  | Parent education level                      |
| final_marks       | Final marks (Target Variable)               |

---

## 🔄 Project Workflow

1. Problem Understanding
2. Data Collection
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Data Preprocessing
6. Feature Engineering
7. Model Training
8. Model Evaluation
9. Prediction
10. Conclusion

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Joblib

### Development Environment

* Jupyter Notebook

---

## 📈 Exploratory Data Analysis (EDA)

The following visualizations were performed:

* Study Hours vs Final Marks
* Final Marks Distribution
* Attendance vs Final Marks
* Correlation Heatmap
* Feature Importance Analysis

---

## 🤖 Machine Learning Models Used

### Linear Regression

A baseline regression model used for predicting student marks.

### Random Forest Regressor

An ensemble learning algorithm that improves prediction accuracy through multiple decision trees.

### XGBoost Regressor

A powerful gradient boosting algorithm used to achieve higher predictive performance.

---

## 📏 Evaluation Metrics

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 📂 Project Structure

Student-Performance-Prediction/

├── data/

│ └── Student_Performance_Dataset.csv

├── notebook/

│ └── Student_Performance_Prediction.ipynb

├── report/

│ └── Student_Performance_Prediction_Report.pdf

├── README.md

└── requirements.txt

---

## 📊 Results

* Data preprocessing and feature engineering improved model performance.
* Study Hours, Attendance, and Previous Score were identified as the most influential features.
* XGBoost achieved the best prediction accuracy among all implemented models.
* The trained model successfully predicts student academic performance based on input features.

---

## 🚀 Future Scope

* Deployment using Streamlit or Flask.
* Integration with educational management systems.
* Real-time student performance monitoring.
* Use of deep learning techniques for enhanced prediction accuracy.

---

## ✅ Conclusion

This project demonstrates the application of Machine Learning in the education domain by predicting student academic performance using historical and behavioral data. The developed models provide valuable insights into factors affecting student success and can support educators in making data-driven decisions.

---

## 📚 References

1. Scikit-Learn Documentation
2. XGBoost Documentation
3. Pandas Documentation
4. NumPy Documentation
5. Machine Learning with Python Resources
