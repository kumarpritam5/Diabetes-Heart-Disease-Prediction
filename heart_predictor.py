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


class HeartDiseasePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

    @staticmethod
    def load_and_preprocess_data(file_path):
        """Load and preprocess the heart disease dataset"""
        try:
            # Get the project root directory and construct full path
            project_root = os.path.dirname(os.path.abspath(__file__))
            datasets_path = os.path.join(project_root, 'datasets')
            full_path = os.path.join(datasets_path, 'heart.csv')

            print(f"📊 Loading Heart Disease Dataset...")
            print(f"Looking for file at: {full_path}")

            # Check if file exists
            if not os.path.exists(full_path):
                print(f"❌ File not found: {full_path}")
                print("Please make sure 'heart.csv' exists in the datasets folder")
                return None

            df = pd.read_csv(full_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📁 Shape: {df.shape}")

            # Handle missing values if any
            if df.isnull().sum().any():
                print("🔄 Handling missing values...")
                df = df.fillna(df.mean())

            return df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def train_model(self, df, target_column='target', feature_mask=None):
        """Train the heart disease prediction model with optional feature selection"""
        if df is None:
            print("❌ No data available for training")
            return False

        try:
            print("🔄 Preparing data for training...")
            X = df.drop(columns=[target_column])
            y = df[target_column]

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

            print("📊 Scaling features...")
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            print("🤖 Training Random Forest model...")
            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ Model trained successfully!")
            print(f"🎯 Accuracy: {accuracy:.4f}")

            self.is_trained = True
            return True

        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False

    def predict_heart_disease(self, features):
        """Predict heart disease for new patient data"""
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
                'risk_level': self._get_risk_level(probability)
            }

        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return None

    @staticmethod
    def _get_risk_level(probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"


def demo_heart_prediction():
    """Demo function to test the heart disease predictor"""
    print("❤️ Heart Disease Prediction Demo")
    print("=" * 40)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Load data
    df = predictor.load_and_preprocess_data('datasets/heart.csv')

    if df is not None:
        # Train model
        success = predictor.train_model(df)

        if success:
            # Example prediction - sample patient data
            print("\n🧪 Example Prediction:")
            example_patient = [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]

            result = predictor.predict_heart_disease(example_patient)

            if result:
                print(f"🎯 Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
                print(f"📊 Probability: {result['probability']:.4f}")
                print(f"⚠️ Risk Level: {result['risk_level']}")


if __name__ == "__main__":
    demo_heart_prediction()