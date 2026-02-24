import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')


class DiabetesPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    @staticmethod
    def load_and_preprocess_data(file_path):
        """Load and preprocess the diabetes dataset"""
        try:
            # Get the project root directory and construct full path
            project_root = os.path.dirname(os.path.abspath(__file__))
            datasets_path = os.path.join(project_root, 'datasets')
            full_path = os.path.join(datasets_path, 'diabetes.csv')

            print(f"📊 Loading Diabetes Dataset...")
            print(f"Looking for file at: {full_path}")

            # Check if file exists
            if not os.path.exists(full_path):
                print(f"❌ File not found: {full_path}")
                print("Please make sure 'diabetes.csv' exists in the datasets folder")
                return None

            data = pd.read_csv(full_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📁 Shape: {data.shape}")

            # Handle missing values
            if data.isnull().sum().any():
                print("🔄 Handling missing values...")
                data = data.fillna(data.mean())

            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def train_model(self, data, target_column='Outcome', feature_mask=None):
        """Train the Random Forest model with optional feature selection"""
        if data is None:
            print("No data available for training")
            return False

        try:
            # Prepare features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Apply feature mask if provided (from Genetic Algorithm)
            if feature_mask is not None:
                selected_features = [self.feature_names[i] for i, selected in enumerate(feature_mask) if selected]
                if selected_features:
                    X = X[selected_features]
                    print(f"Training with GA-selected features: {selected_features}")

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)

            # Calculate and display metrics
            model_accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ Model trained successfully!")
            print(f"🎯 Accuracy: {model_accuracy:.4f}")

            self.is_trained = True
            return True

        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False

    def predict_diabetes(self, features):
        """Predict diabetes probability for new data"""
        if not self.is_trained:
            print("Model not trained yet. Please train the model first.")
            return None

        try:
            # Ensure features are in correct format
            if isinstance(features, list):
                features = np.array(features).reshape(1, -1)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]

            # Convert numpy types to native Python types
            prediction = int(prediction)
            probability = float(probability[1])

            return {
                'prediction': prediction,
                'probability': probability,
                'risk_level': self.get_risk_level(probability)
            }

        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return None

    @staticmethod
    def get_risk_level(probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"


def main():
    """Main function to run the diabetes predictor"""
    # Initialize predictor
    predictor = DiabetesPredictor()

    # Load data
    data = predictor.load_and_preprocess_data('datasets/diabetes.csv')

    if data is not None:
        # Train model
        success = predictor.train_model(data)

        if success:
            # Example prediction
            example_features = [2, 120, 70, 20, 79, 25.0, 0.5, 30]
            result = predictor.predict_diabetes(example_features)

            if result:
                print(f"\n🎯 Prediction Result:")
                print(f"Diabetes: {'Yes' if result['prediction'] == 1 else 'No'}")
                print(f"Probability: {result['probability']:.4f}")
                print(f"Risk Level: {result['risk_level']}")


if __name__ == "__main__":
    main()