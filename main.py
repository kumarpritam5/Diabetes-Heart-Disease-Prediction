from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
import json
import random
from typing import List, Tuple, Dict, Any

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

app = Flask(__name__, template_folder='templates')

# Global variables
diabetes_predictor = None
heart_predictor = None
diabetes_trained = False
heart_trained = False


# ==============================
# Genetic Algorithm (KEPT)
# ==============================

class GeneticAlgorithmOptimizer:
    def __init__(self, population_size=30, generations=50, crossover_rate=0.8, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.best_individual = None
        self.best_fitness = float('-inf')

    def initialize_population(self, chromosome_length: int):
        return [[random.random() > 0.5 for _ in range(chromosome_length)]
                for _ in range(self.population_size)]

    def fitness_function(self, chromosome, X, y, predictor, feature_names):
        from sklearn.model_selection import train_test_split

        selected_indices = [i for i, selected in enumerate(chromosome) if selected]
        if not selected_indices:
            return 0.0

        X_selected = X[:, selected_indices]
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )

        predictor.model.fit(X_train, y_train)
        return predictor.model.score(X_test, y_test)


# ==============================
# Fuzzy Logic (KEPT)
# ==============================

class FuzzyRiskAssessor:
    def triangular_mf(self, x, a, b, c):
        if x <= a:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return (c - x) / (c - b)
        return 0.0

    def fuzzy_risk_assessment(self, probability, age, comorbidities):

        prob_low = self.triangular_mf(probability, 0, 0, 0.4)
        prob_medium = self.triangular_mf(probability, 0.2, 0.5, 0.8)
        prob_high = self.triangular_mf(probability, 0.6, 1.0, 1.0)

        age_young = self.triangular_mf(age, 0, 0, 45)
        age_middle = self.triangular_mf(age, 35, 50, 65)
        age_senior = self.triangular_mf(age, 55, 70, 100)

        comorb_low = self.triangular_mf(comorbidities, 0, 0, 2)
        comorb_medium = self.triangular_mf(comorbidities, 1, 3, 5)
        comorb_high = self.triangular_mf(comorbidities, 4, 6, 10)

        risk_low = min(prob_low, age_young, comorb_low)
        risk_medium = max(
            min(prob_medium, age_middle, comorb_medium),
            min(prob_low, age_senior, comorb_medium)
        )
        risk_high = max(
            min(prob_high, age_senior, comorb_high),
            min(prob_medium, age_senior, comorb_high)
        )

        total_risk = risk_low * 0.25 + risk_medium * 0.5 + risk_high * 0.75
        total_membership = risk_low + risk_medium + risk_high

        fuzzy_risk = total_risk / total_membership if total_membership > 0 else 0.5

        if fuzzy_risk < 0.4:
            level = "Low"
        elif fuzzy_risk < 0.7:
            level = "Medium"
        else:
            level = "High"

        return {
            "fuzzy_risk_score": fuzzy_risk,
            "risk_level": level
        }


# ==============================
# Initialize Models (KEPT)
# ==============================

def initialize_models():
    global diabetes_predictor, heart_predictor
    global diabetes_trained, heart_trained

    from diabetes_predictor import DiabetesPredictor
    from heart_predictor import HeartDiseasePredictor

    diabetes_predictor = DiabetesPredictor()
    heart_predictor = HeartDiseasePredictor()

    diabetes_data = diabetes_predictor.load_and_preprocess_data('datasets/diabetes.csv')
    heart_data = heart_predictor.load_and_preprocess_data('datasets/heart.csv')

    if diabetes_data is not None:
        diabetes_trained = diabetes_predictor.train_model(diabetes_data)

    if heart_data is not None:
        heart_trained = heart_predictor.train_model(heart_data)


# ==============================
# Routes (ONLY REQUIRED ONES)
# ==============================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if not diabetes_trained:
        return jsonify({'error': 'Model not trained'}), 500

    data = request.json

    features = [
        float(data.get('pregnancies', 0)),
        float(data.get('glucose', 0)),
        float(data.get('blood_pressure', 0)),
        float(data.get('skin_thickness', 0)),
        float(data.get('insulin', 0)),
        float(data.get('bmi', 0)),
        float(data.get('dpf', 0)),
        float(data.get('age', 0))
    ]

    result = diabetes_predictor.predict_diabetes(features)

    fuzzy = FuzzyRiskAssessor().fuzzy_risk_assessment(
        result['probability'],
        float(data.get('age', 0)),
        float(data.get('pregnancies', 0))
    )

    result['enhanced_risk_score'] = fuzzy['fuzzy_risk_score']
    result['enhanced_risk_level'] = fuzzy['risk_level']

    return jsonify(result)


@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if not heart_trained:
        return jsonify({'error': 'Model not trained'}), 500

    data = request.json

    features = [
        float(data.get('age', 0)),
        float(data.get('sex', 0)),
        float(data.get('cp', 0)),
        float(data.get('trestbps', 0)),
        float(data.get('chol', 0)),
        float(data.get('fbs', 0)),
        float(data.get('restecg', 0)),
        float(data.get('thalach', 0)),
        float(data.get('exang', 0)),
        float(data.get('oldpeak', 0)),
        float(data.get('slope', 0)),
        float(data.get('ca', 0)),
        float(data.get('thal', 0))
    ]

    result = heart_predictor.predict_heart_disease(features)

    fuzzy = FuzzyRiskAssessor().fuzzy_risk_assessment(
        result['probability'],
        float(data.get('age', 0)),
        float(data.get('ca', 0))
    )

    result['enhanced_risk_score'] = fuzzy['fuzzy_risk_score']
    result['enhanced_risk_level'] = fuzzy['risk_level']

    return jsonify(result)


# ==============================
# Run Server
# ==============================

if __name__ == '__main__':
    initialize_models()
    app.run(debug=False, host='0.0.0.0', port=5000)