"""
Data Preprocessing Module for Salary Prediction Project

This module handles data cleaning, missing value imputation,
and basic data transformations.

Author: [Your Name]
Date: July 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for salary prediction
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputers = {}
        
    def load_data(self, filepath):
        """
        Load and perform initial data validation
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def basic_info(self, df):
        """
        Display basic information about the dataset
        
        Args:
            df (pd.DataFrame): Input dataset
        """
        print("=== DATASET OVERVIEW ===")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\n=== COLUMN INFORMATION ===")
        print(df.info())
        print("\n=== MISSING VALUES ===")
        missing_info = df.isnull().sum()
        missing_info = missing_info[missing_info > 0]
        if len(missing_info) > 0:
            print(missing_info)
        else:
            print("No missing values found")
            
    def handle_missing_values(self, df):
        """
        Handle missing values using appropriate strategies
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with handled missing values
        """
        df_clean = df.copy()
        
        # Handle missing values for numerical columns
        numerical_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if df_clean[col].isnull().sum() > 0:
                if col in ['Age', 'Years of Experience']:
                    # Use KNN imputation for Age and Experience
                    imputer = KNNImputer(n_neighbors=5)
                    df_clean[col] = imputer.fit_transform(df_clean[[col]])
                else:
                    # Use median for other numerical columns
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Handle missing values for categorical columns
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_clean[col].isnull().sum() > 0:
                # Fill with mode (most frequent value)
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_value, inplace=True)
        
        logger.info("Missing values handled successfully")
        return df_clean
    
    def detect_outliers(self, df, columns=None):
        """
        Detect outliers using IQR method
        
        Args:
            df (pd.DataFrame): Input dataset
            columns (list): List of columns to check for outliers
            
        Returns:
            dict: Dictionary with outlier information
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return outlier_info
    
    def handle_outliers(self, df, method='cap', threshold=0.05):
        """
        Handle outliers using specified method
        
        Args:
            df (pd.DataFrame): Input dataset
            method (str): Method to handle outliers ('cap', 'remove', 'log')
            threshold (float): Threshold for outlier treatment
            
        Returns:
            pd.DataFrame: Dataset with handled outliers
        """
        df_clean = df.copy()
        
        # Focus on salary outliers
        if 'Salary' in df_clean.columns:
            Q1 = df_clean['Salary'].quantile(0.25)
            Q3 = df_clean['Salary'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                df_clean['Salary'] = np.where(df_clean['Salary'] > upper_bound, 
                                            upper_bound, df_clean['Salary'])
                df_clean['Salary'] = np.where(df_clean['Salary'] < lower_bound, 
                                            lower_bound, df_clean['Salary'])
            elif method == 'remove':
                df_clean = df_clean[(df_clean['Salary'] >= lower_bound) & 
                                  (df_clean['Salary'] <= upper_bound)]
            elif method == 'log':
                df_clean['Salary'] = np.log1p(df_clean['Salary'])
        
        logger.info(f"Outliers handled using {method} method")
        return df_clean
    
    def standardize_categories(self, df):
        """
        Standardize categorical variables
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with standardized categories
        """
        df_clean = df.copy()
        
        # Standardize education levels
        if 'Education Level' in df_clean.columns:
            education_mapping = {
                'High School': 'High School',
                "Bachelor's": 'Bachelor',
                "Bachelor's Degree": 'Bachelor',
                "Master's": 'Master',
                "Master's Degree": 'Master',
                'PhD': 'PhD',
                'phD': 'PhD'
            }
            df_clean['Education Level'] = df_clean['Education Level'].map(education_mapping)
            df_clean['Education Level'] = df_clean['Education Level'].fillna('Bachelor')
        
        # Clean job titles (remove extra spaces, standardize case)
        if 'Job Title' in df_clean.columns:
            df_clean['Job Title'] = df_clean['Job Title'].str.strip().str.title()
        
        # Standardize gender
        if 'Gender' in df_clean.columns:
            df_clean['Gender'] = df_clean['Gender'].str.strip().str.title()
        
        # Standardize race
        if 'Race' in df_clean.columns:
            df_clean['Race'] = df_clean['Race'].str.strip().str.title()
        
        logger.info("Categories standardized successfully")
        return df_clean
    
    def create_derived_features(self, df):
        """
        Create additional features based on existing ones
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with derived features
        """
        df_enhanced = df.copy()
        
        # Experience categories
        if 'Years of Experience' in df_enhanced.columns:
            df_enhanced['Experience_Category'] = pd.cut(
                df_enhanced['Years of Experience'],
                bins=[0, 2, 5, 10, 20, 50],
                labels=['Entry', 'Junior', 'Mid', 'Senior', 'Expert'],
                include_lowest=True
            )
        
        # Age categories
        if 'Age' in df_enhanced.columns:
            df_enhanced['Age_Category'] = pd.cut(
                df_enhanced['Age'],
                bins=[0, 25, 35, 45, 55, 70],
                labels=['Young', 'Early_Career', 'Mid_Career', 'Senior_Career', 'Late_Career'],
                include_lowest=True
            )
        
        # Job level categorization
        if 'Job Title' in df_enhanced.columns:
            senior_keywords = ['senior', 'lead', 'principal', 'director', 'manager', 'head']
            df_enhanced['Job_Level'] = df_enhanced['Job Title'].str.lower().apply(
                lambda x: 'Senior' if any(keyword in x for keyword in senior_keywords) else 'Junior'
            )
        
        logger.info("Derived features created successfully")
        return df_enhanced
    
    def preprocess_pipeline(self, df):
        """
        Complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Fully preprocessed dataset
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Basic info
        self.basic_info(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Standardize categories
        df = self.standardize_categories(df)
        
        # Step 4: Create derived features
        df = self.create_derived_features(df)
        
        # Step 5: Handle outliers
        df = self.handle_outliers(df, method='cap')
        
        logger.info("Preprocessing pipeline completed successfully")
        return df

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data("data/raw/salary_data.csv")
    df_processed = preprocessor.preprocess_pipeline(df)
    
    # Save processed data
    df_processed.to_csv("data/processed/salary_data_processed.csv", index=False)
    print("Processed data saved successfully!")
