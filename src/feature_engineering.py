"""
Feature Engineering Module for Salary Prediction Project

This module handles advanced feature engineering techniques including
encoding, scaling, and feature selection.

Author: [Your Name]
Date: July 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler,
    PolynomialFeatures, PowerTransformer
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor

# Optional advanced encoders
try:
    from category_encoders import TargetEncoder, BinaryEncoder
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    print("⚠️  category_encoders not available. Using basic encoding methods.")
    CATEGORY_ENCODERS_AVAILABLE = False
    
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering class for salary prediction
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_selector = None
        self.selected_features = None
        
    def encode_categorical_features(self, df, target_column='Salary', method='auto'):
        """
        Encode categorical features using various methods
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            method (str): Encoding method ('onehot', 'label', 'target', 'auto')
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]
        
        for col in categorical_columns:
            unique_values = df_encoded[col].nunique()
            
            if method == 'auto':
                # Choose encoding method based on cardinality
                if unique_values <= 5:
                    encoding_method = 'onehot'
                elif unique_values <= 20:
                    encoding_method = 'target'
                else:
                    encoding_method = 'label'
            else:
                encoding_method = method
            
            if encoding_method == 'onehot':
                # One-hot encoding for low cardinality
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(df_encoded[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                # Create DataFrame for encoded features
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df_encoded.index)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_df], axis=1)
                self.encoders[col] = encoder
                
            elif encoding_method == 'target':
                # Target encoding for medium cardinality
                if CATEGORY_ENCODERS_AVAILABLE:
                    encoder = TargetEncoder()
                    df_encoded[f"{col}_target_encoded"] = encoder.fit_transform(
                        df_encoded[col], df_encoded[target_column]
                    )
                    self.encoders[col] = encoder
                else:
                    # Fallback: Manual target encoding
                    target_means = df_encoded.groupby(col)[target_column].mean()
                    df_encoded[f"{col}_target_encoded"] = df_encoded[col].map(target_means)
                    self.encoders[col] = target_means
                df_encoded.drop(col, axis=1, inplace=True)
                
            elif encoding_method == 'label':
                # Label encoding for high cardinality
                encoder = LabelEncoder()
                df_encoded[f"{col}_label_encoded"] = encoder.fit_transform(df_encoded[col])
                df_encoded.drop(col, axis=1, inplace=True)
                self.encoders[col] = encoder
        
        logger.info(f"Categorical encoding completed. Methods used: {method}")
        return df_encoded
    
    def scale_numerical_features(self, df, target_column='Salary', method='standard'):
        """
        Scale numerical features
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            method (str): Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: Dataset with scaled numerical features
        """
        df_scaled = df.copy()
        numerical_columns = df_scaled.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in numerical_columns if col != target_column]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method")
        
        df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
        self.scalers['numerical'] = scaler
        
        logger.info(f"Numerical features scaled using {method} method")
        return df_scaled
    
    def create_polynomial_features(self, df, target_column='Salary', degree=2, 
                                 include_columns=None):
        """
        Create polynomial features for numerical variables
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            degree (int): Polynomial degree
            include_columns (list): Specific columns to include
            
        Returns:
            pd.DataFrame: Dataset with polynomial features
        """
        df_poly = df.copy()
        
        if include_columns is None:
            # Use key numerical features
            include_columns = ['Age', 'Years of Experience']
            include_columns = [col for col in include_columns if col in df_poly.columns]
        
        if include_columns:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(df_poly[include_columns])
            
            # Create feature names
            feature_names = poly.get_feature_names_out(include_columns)
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_poly.index)
            
            # Remove original columns to avoid duplication
            df_poly = df_poly.drop(include_columns, axis=1)
            df_poly = pd.concat([df_poly, poly_df], axis=1)
            
            self.encoders['polynomial'] = poly
            logger.info(f"Polynomial features created with degree {degree}")
        
        return df_poly
    
    def create_interaction_features(self, df, target_column='Salary'):
        """
        Create interaction features between key variables
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            
        Returns:
            pd.DataFrame: Dataset with interaction features
        """
        df_interaction = df.copy()
        
        # Age and Experience interaction
        if 'Age' in df_interaction.columns and 'Years of Experience' in df_interaction.columns:
            df_interaction['Age_Experience_Ratio'] = df_interaction['Age'] / (df_interaction['Years of Experience'] + 1)
            df_interaction['Age_Experience_Product'] = df_interaction['Age'] * df_interaction['Years of Experience']
        
        # Education and Experience interaction
        education_cols = [col for col in df_interaction.columns if 'Education' in col or 'education' in col.lower()]
        if education_cols and 'Years of Experience' in df_interaction.columns:
            for edu_col in education_cols:
                if df_interaction[edu_col].dtype in [np.number]:
                    df_interaction[f'{edu_col}_Experience_Product'] = (
                        df_interaction[edu_col] * df_interaction['Years of Experience']
                    )
        
        logger.info("Interaction features created successfully")
        return df_interaction
    
    def feature_selection(self, df, target_column='Salary', method='recursive', k=20):
        """
        Select most important features
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            method (str): Selection method ('univariate', 'recursive', 'model_based')
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Dataset with selected features
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        if method == 'univariate':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'recursive':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=k)
        elif method == 'model_based':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator=estimator, max_features=k)
        else:
            raise ValueError("Unsupported feature selection method")
        
        X_selected = selector.fit_transform(X, y)
        selected_feature_names = X.columns[selector.get_support()].tolist()
        
        df_selected = pd.DataFrame(X_selected, columns=selected_feature_names, index=df.index)
        df_selected[target_column] = y
        
        self.feature_selector = selector
        self.selected_features = selected_feature_names
        
        logger.info(f"Feature selection completed. Selected {len(selected_feature_names)} features using {method} method")
        return df_selected
    
    def transform_target_variable(self, df, target_column='Salary', method='log'):
        """
        Transform target variable for better model performance
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            method (str): Transformation method ('log', 'sqrt', 'boxcox')
            
        Returns:
            pd.DataFrame: Dataset with transformed target
        """
        df_transformed = df.copy()
        
        if method == 'log':
            df_transformed[f"{target_column}_log"] = np.log1p(df_transformed[target_column])
        elif method == 'sqrt':
            df_transformed[f"{target_column}_sqrt"] = np.sqrt(df_transformed[target_column])
        elif method == 'boxcox':
            transformer = PowerTransformer(method='box-cox')
            df_transformed[f"{target_column}_boxcox"] = transformer.fit_transform(
                df_transformed[[target_column]].values
            )
            self.scalers['target_transformer'] = transformer
        
        logger.info(f"Target variable transformed using {method} method")
        return df_transformed
    
    def create_binned_features(self, df, target_column='Salary'):
        """
        Create binned versions of continuous variables
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            
        Returns:
            pd.DataFrame: Dataset with binned features
        """
        df_binned = df.copy()
        
        # Bin Age
        if 'Age' in df_binned.columns:
            df_binned['Age_Binned'] = pd.cut(
                df_binned['Age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['<25', '25-35', '35-45', '45-55', '55+']
            )
        
        # Bin Experience
        if 'Years of Experience' in df_binned.columns:
            df_binned['Experience_Binned'] = pd.cut(
                df_binned['Years of Experience'],
                bins=[0, 2, 5, 10, 20, 50],
                labels=['0-2', '2-5', '5-10', '10-20', '20+']
            )
        
        logger.info("Binned features created successfully")
        return df_binned
    
    def feature_engineering_pipeline(self, df, target_column='Salary', 
                                   encoding_method='auto', scaling_method='standard',
                                   create_polynomials=True, create_interactions=True,
                                   feature_selection_method='recursive', n_features=30):
        """
        Complete feature engineering pipeline
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            encoding_method (str): Categorical encoding method
            scaling_method (str): Numerical scaling method
            create_polynomials (bool): Whether to create polynomial features
            create_interactions (bool): Whether to create interaction features
            feature_selection_method (str): Feature selection method
            n_features (int): Number of features to select
            
        Returns:
            pd.DataFrame: Fully engineered dataset
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Step 1: Encode categorical features
        df = self.encode_categorical_features(df, target_column, encoding_method)
        
        # Step 2: Create binned features
        df = self.create_binned_features(df, target_column)
        
        # Step 3: Create polynomial features
        if create_polynomials:
            df = self.create_polynomial_features(df, target_column)
        
        # Step 4: Create interaction features
        if create_interactions:
            df = self.create_interaction_features(df, target_column)
        
        # Step 5: Scale numerical features
        df = self.scale_numerical_features(df, target_column, scaling_method)
        
        # Step 6: Feature selection
        df = self.feature_selection(df, target_column, feature_selection_method, n_features)
        
        logger.info("Feature engineering pipeline completed successfully")
        return df
    
    def get_feature_importance(self, df, target_column='Salary'):
        """
        Calculate feature importance using Random Forest
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            
        Returns:
            pd.DataFrame: Feature importance rankings
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    # Load preprocessed data
    df = pd.read_csv("data/processed/salary_data_processed.csv")
    
    # Apply feature engineering
    df_engineered = engineer.feature_engineering_pipeline(df)
    
    # Get feature importance
    importance = engineer.get_feature_importance(df_engineered)
    print("Top 10 Important Features:")
    print(importance.head(10))
    
    # Save engineered data
    df_engineered.to_csv("data/processed/salary_data_engineered.csv", index=False)
    print("Engineered data saved successfully!")