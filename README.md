# Diabetes-Heart-Disease-Prediction
This project is a Machine Learning-based medical prediction system that predicts Diabetes and Heart Disease using supervised learning techniques. The main objective of this project is to build an intelligent and interpretable medical prediction system by combining Machine Learning with optimization and fuzzy logic techniques.

--> How Diabetes Prediction Works
For diabetes prediction, I used:
1.Random Forest Classifier as the main prediction model.
2.Medical dataset preprocessing (handling missing values and scaling features).
3.Train-test split (80/20) to evaluate model performance.
4.Feature scaling using StandardScaler for better accuracy.
5.Probability-based output to measure risk level.

Additionally:
Genetic Algorithm was used for feature selection to improve model performance.
Fuzzy Logic was applied on prediction probability to convert it into meaningful risk levels (Low / Medium / High).

--> How Heart Disease Prediction Works:- 
For heart disease prediction, I implemented:
1.Random Forest Classifier (100 decision trees).
2.Data preprocessing and missing value handling.
3.Stratified train-test split to maintain class balance.
4.Feature scaling using StandardScaler.
5.Probability prediction using predict_proba().

The probability is then categorized into:
Low Risk
Medium Risk
High Risk
This makes the output easier to understand instead of just giving 0 or 1.

--> Methods Used in This Project
1.Supervised Machine Learning
2.Random Forest Algorithm
3.Feature Scaling (Standardization)
4.Train-Test Split Validation
5.Genetic Algorithm (Feature Optimization – Diabetes Module)
6.Fuzzy Logic (Risk Enhancement – Diabetes Module)
7.Probability-Based Risk Classification
