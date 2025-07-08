# Salary Prediction Project Configuration

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
