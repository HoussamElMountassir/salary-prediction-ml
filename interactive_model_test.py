"""
Interactive Model Testing
Allows you to input custom values and get salary predictions
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_model():
    """Load the trained model"""
    model_paths = [
        'models/trained_models/best_model.joblib',
        'models/trained_models/minimal_salary_model.joblib',
        'models/trained_models/basic_model.joblib'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                print(f"✅ Loaded model from: {path}")
                return model
            except Exception as e:
                continue
    
    print("❌ No trained model found!")
    return None

def get_sample_features():
    """Get sample features from the dataset"""
    try:
        # Try processed data first
        if os.path.exists('data/processed/salary_data_processed.csv'):
            df = pd.read_csv('data/processed/salary_data_processed.csv')
        else:
            df = pd.read_csv('Salary_Data_Based_country_and_race.csv')
        
        # Clean and encode the data
        df = df.dropna()
        
        # Encode categorical variables
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'Salary':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        X = df.drop('Salary', axis=1)
        return X.iloc[0], X.columns.tolist()  # Return first row as template
        
    except Exception as e:
        print(f"Error loading sample features: {e}")
        return None, None

def interactive_prediction():
    """Interactive prediction interface"""
    
    print("🎯 INTERACTIVE SALARY PREDICTION")
    print("=" * 40)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Get sample features
    sample_features, feature_names = get_sample_features()
    if sample_features is None:
        return
    
    print(f"\n📊 Model expects {len(feature_names)} features:")
    for i, name in enumerate(feature_names):
        print(f"   {i+1:2d}. {name}")
    
    print("\n🧪 PREDICTION SCENARIOS:")
    print("Choose a scenario to test:")
    print("1. Entry Level Employee")
    print("2. Mid-Level Professional") 
    print("3. Senior Executive")
    print("4. Custom Input")
    print("5. Batch Test (5 random samples)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        # Entry level scenario
        features = sample_features.copy()
        if 'Age' in feature_names:
            features[feature_names.index('Age')] = 25
        if 'Years of Experience' in feature_names:
            features[feature_names.index('Years of Experience')] = 2
        
        prediction = model.predict([features])[0]
        print(f"\n💼 Entry Level Prediction: ${prediction:,.0f}")
        print("   Scenario: Age 25, 2 years experience")
        
    elif choice == "2":
        # Mid-level scenario
        features = sample_features.copy()
        if 'Age' in feature_names:
            features[feature_names.index('Age')] = 35
        if 'Years of Experience' in feature_names:
            features[feature_names.index('Years of Experience')] = 8
        
        prediction = model.predict([features])[0]
        print(f"\n💼 Mid-Level Prediction: ${prediction:,.0f}")
        print("   Scenario: Age 35, 8 years experience")
        
    elif choice == "3":
        # Senior scenario
        features = sample_features.copy()
        if 'Age' in feature_names:
            features[feature_names.index('Age')] = 45
        if 'Years of Experience' in feature_names:
            features[feature_names.index('Years of Experience')] = 20
        
        prediction = model.predict([features])[0]
        print(f"\n💼 Senior Executive Prediction: ${prediction:,.0f}")
        print("   Scenario: Age 45, 20 years experience")
        
    elif choice == "4":
        # Custom input
        print("\n📝 Enter custom values (press Enter to use default):")
        features = sample_features.copy()
        
        for i, name in enumerate(feature_names):
            try:
                current_value = features[i]
                user_input = input(f"   {name} (current: {current_value}): ").strip()
                if user_input:
                    features[i] = float(user_input)
            except:
                pass
        
        prediction = model.predict([features])[0]
        print(f"\n💼 Custom Prediction: ${prediction:,.0f}")
        
    elif choice == "5":
        # Batch test
        print("\n🔬 BATCH TESTING (5 random samples):")
        
        # Load test data
        try:
            df = pd.read_csv('Salary_Data_Based_country_and_race.csv')
            df = df.dropna()
            
            # Encode categorical variables
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'Salary':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
            
            # Random samples
            sample_indices = np.random.choice(len(df), 5, replace=False)
            
            for i, idx in enumerate(sample_indices):
                X_sample = df.drop('Salary', axis=1).iloc[idx:idx+1]
                y_actual = df['Salary'].iloc[idx]
                y_pred = model.predict(X_sample)[0]
                error = abs(y_actual - y_pred)
                error_pct = (error / y_actual) * 100
                
                print(f"\n   Sample {i+1}:")
                print(f"   Actual: ${y_actual:,.0f}")
                print(f"   Predicted: ${y_pred:,.0f}")
                print(f"   Error: ${error:,.0f} ({error_pct:.1f}%)")
                
        except Exception as e:
            print(f"   ❌ Batch test failed: {e}")
    
    else:
        print("❌ Invalid choice")

def model_info():
    """Display model information"""
    
    print("\n📋 MODEL INFORMATION:")
    print("-" * 25)
    
    model = load_model()
    if model is None:
        return
    
    print(f"Model Type: {type(model).__name__}")
    
    if hasattr(model, 'n_estimators'):
        print(f"Number of Estimators: {model.n_estimators}")
    
    if hasattr(model, 'feature_importances_'):
        print(f"Number of Features: {len(model.feature_importances_)}")
        
        # Show top 5 important features
        _, feature_names = get_sample_features()
        if feature_names:
            importance_pairs = list(zip(feature_names, model.feature_importances_))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print("\nTop 5 Important Features:")
            for i, (name, importance) in enumerate(importance_pairs[:5]):
                print(f"   {i+1}. {name}: {importance:.4f}")
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print(f"\nKey Parameters:")
        key_params = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']
        for param in key_params:
            if param in params:
                print(f"   {param}: {params[param]}")

def performance_summary():
    """Show model performance summary"""
    
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 25)
    
    # Check for performance reports
    report_files = [
        'reports/model_performance_minimal.csv',
        'reports/MODEL_TEST_REPORT.md',
        'reports/FINAL_REPORT.md'
    ]
    
    for report_file in report_files:
        if os.path.exists(report_file):
            print(f"✅ Found report: {report_file}")
            
            if report_file.endswith('.csv'):
                try:
                    df = pd.read_csv(report_file)
                    if 'r2_score' in df.columns:
                        r2 = df['r2_score'].iloc[0]
                        print(f"   R² Score: {r2:.4f} ({r2*100:.1f}% accuracy)")
                    if 'rmse' in df.columns:
                        rmse = df['rmse'].iloc[0]
                        print(f"   RMSE: ${rmse:,.2f}")
                    if 'mae' in df.columns:
                        mae = df['mae'].iloc[0]
                        print(f"   MAE: ${mae:,.2f}")
                except:
                    pass
            break
    else:
        print("⚠️ No performance reports found")
        print("   Run 'python test_model.py' to generate performance metrics")

def main_menu():
    """Main menu for interactive testing"""
    
    while True:
        print("\n" + "="*50)
        print("🧪 MODEL TESTING MENU")
        print("="*50)
        print("1. 🎯 Make Salary Predictions")
        print("2. 📋 View Model Information") 
        print("3. 📊 View Performance Summary")
        print("4. 🔬 Run Full Test Suite")
        print("5. ❌ Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            interactive_prediction()
        elif choice == "2":
            model_info()
        elif choice == "3":
            performance_summary()
        elif choice == "4":
            print("\n🔬 Running comprehensive test suite...")
            os.system("python test_model.py")
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("🎯 INTERACTIVE MODEL TESTING SUITE")
    print("Welcome to the Salary Prediction Model Tester!")
    
    # Quick model check
    model = load_model()
    if model is None:
        print("\n❌ No trained model found!")
        print("Please run one of these first:")
        print("   python main.py")
        print("   python main_patched.py") 
        print("   python minimal_salary_prediction.py")
        exit()
    
    print("✅ Model loaded successfully!")
    main_menu()
