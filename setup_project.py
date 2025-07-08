"""
Project Setup Script for Salary Prediction Project

This script sets up the complete project environment and runs
initial data validation.

Author: [Your Name]
Date: July 2025
"""

import os
import shutil
import subprocess
import sys

def create_project_structure():
    """Create the complete project directory structure"""
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'notebooks',
        'src',
        'models/trained_models',
        'models/model_artifacts',
        'reports/figures',
        'logs',
        'tests',
        'config'
    ]
    
    print("📁 Creating project directory structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ Created: {directory}/")
    
    print("✅ Project structure created successfully!")

def copy_dataset():
    """Copy the salary dataset to the appropriate location"""
    
    source_file = "Salary_Data_Based_country_and_race.csv"
    destination = "data/raw/Salary_Data_Based_country_and_race.csv"
    
    if os.path.exists(source_file):
        shutil.copy2(source_file, destination)
        print(f"📊 Dataset copied to {destination}")
        return True
    else:
        print(f"⚠️  Dataset not found: {source_file}")
        print("   Please ensure the CSV file is in the project root directory")
        return False

def install_dependencies():
    """Install required Python packages"""
    
    print("📦 Installing required dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing dependencies")
        print("   Please run: pip install -r requirements.txt manually")
        return False

def validate_setup():
    """Validate the project setup"""
    
    print("🔍 Validating project setup...")
    
    # Check critical files
    critical_files = [
        'requirements.txt',
        'main.py',
        'src/data_preprocessing.py',
        'src/feature_engineering.py', 
        'src/model_training.py',
        'src/model_evaluation.py',
        'notebooks/01_exploratory_data_analysis.ipynb'
    ]
    
    missing_files = []
    for file_path in critical_files:
        if os.path.exists(file_path):
            print(f"   ✓ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    # Check dataset
    dataset_path = "data/raw/Salary_Data_Based_country_and_race.csv"
    if os.path.exists(dataset_path):
        print(f"   ✓ {dataset_path}")
        
        # Quick dataset validation
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"   📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"   ⚠️  Dataset validation failed: {str(e)}")
    else:
        print(f"   ❌ {dataset_path}")
        missing_files.append(dataset_path)
    
    if missing_files:
        print(f"\n❌ Setup incomplete. Missing files: {missing_files}")
        return False
    else:
        print("\n✅ Project setup validation passed!")
        return True

def create_config_file():
    """Create a configuration file for the project"""
    
    config_content = """# Salary Prediction Project Configuration

# Data Configuration
DATA_PATH = "data/raw/Salary_Data_Based_country_and_race.csv"
TARGET_COLUMN = "Salary"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature Engineering
ENCODING_METHOD = "auto"
SCALING_METHOD = "standard"
N_FEATURES = 25
CREATE_POLYNOMIALS = True
CREATE_INTERACTIONS = True

# Model Training
MODELS_TO_TUNE = ["random_forest", "xgboost", "lightgbm", "catboost"]
SEARCH_METHOD = "random"
CV_FOLDS = 5

# Output Paths
PROCESSED_DATA_PATH = "data/processed/"
MODEL_SAVE_PATH = "models/trained_models/"
REPORTS_PATH = "reports/"
"""
    
    with open("config/config.py", "w") as f:
        f.write(config_content)
    
    print("⚙️  Configuration file created: config/config.py")

def print_next_steps():
    """Print instructions for next steps"""
    
    print("\n" + "="*60)
    print("🎉 PROJECT SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. 🚀 Run the complete pipeline:")
    print("   python main.py")
    
    print("\n2. 📊 Or run individual components:")
    print("   - Exploratory Data Analysis: Open notebooks/01_exploratory_data_analysis.ipynb")
    print("   - Data Preprocessing: python -c \"from src.data_preprocessing import *\"")
    print("   - Model Training: python -c \"from src.model_training import *\"")
    
    print("\n3. 📈 View results:")
    print("   - Check reports/ directory for analysis")
    print("   - Best model saved in models/trained_models/")
    print("   - Logs available in logs/ directory")
    
    print("\n4. 🎯 For recruiters and managers:")
    print("   - README.md: Project overview and achievements")
    print("   - reports/FINAL_REPORT.md: Comprehensive results")
    print("   - notebooks/: Interactive analysis")
    
    print("\n💡 TIP: This project demonstrates:")
    print("   ✓ End-to-end ML pipeline development")
    print("   ✓ Advanced feature engineering")
    print("   ✓ Model selection and hyperparameter tuning")
    print("   ✓ Comprehensive model evaluation")
    print("   ✓ Bias analysis and ethical AI considerations")
    print("   ✓ Production-ready code structure")

def main():
    """Main setup function"""
    
    print("🎯 SALARY PREDICTION PROJECT SETUP")
    print("="*50)
    
    # Step 1: Create project structure
    create_project_structure()
    
    # Step 2: Copy dataset
    dataset_copied = copy_dataset()
    
    # Step 3: Create configuration
    create_config_file()
    
    # Step 4: Install dependencies (optional)
    print("\n📦 Would you like to install dependencies? (y/n): ", end="")
    install_deps = input().lower().strip() == 'y'
    
    if install_deps:
        install_dependencies()
    
    # Step 5: Validate setup
    print()
    setup_valid = validate_setup()
    
    # Step 6: Print next steps
    if setup_valid and dataset_copied:
        print_next_steps()
    else:
        print("\n⚠️  Setup completed with warnings. Please address missing components.")

if __name__ == "__main__":
    main()