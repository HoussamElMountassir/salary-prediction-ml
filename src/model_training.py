"""
Model Training Module for Salary Prediction Project

This module handles training multiple machine learning models
and hyperparameter optimization.

Author: [Your Name]
Date: July 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score, validation_curve
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive model training class for salary prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_params = None
        self.results = {}
        
    def initialize_models(self):
        """Initialize all models with default parameters"""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state),
            'decision_tree': DecisionTreeRegressor(random_state=self.random_state),
            'random_forest': RandomForestRegressor(random_state=self.random_state),
            'extra_trees': ExtraTreesRegressor(random_state=self.random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'ada_boost': AdaBoostRegressor(random_state=self.random_state),
            'xgboost': xgb.XGBRegressor(random_state=self.random_state, eval_metric='rmse'),
            'lightgbm': lgb.LGBMRegressor(random_state=self.random_state, verbose=-1),
            'catboost': CatBoostRegressor(random_state=self.random_state, verbose=False),
            'svr': SVR(),
            'knn': KNeighborsRegressor()
        }
        logger.info(f"Initialized {len(self.models)} models")
    
    def get_hyperparameter_grids(self):
        """Define hyperparameter grids for each model"""
        param_grids = {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky']
            },
            'lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 2000, 3000]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            },
            'catboost': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 6, 9],
                'l2_leaf_reg': [1, 3, 5]
            }
        }
        return param_grids
    
    def split_data(self, df, target_column='Salary', test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets"""
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_baseline_models(self, X_train, y_train, X_val, y_val):
        """Train all models with default parameters"""
        self.initialize_models()
        baseline_results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                baseline_results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                logger.info(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                baseline_results[name] = {'error': str(e)}
        
        self.results['baseline'] = baseline_results
        return baseline_results
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, 
                            models_to_tune=None, search_method='grid'):
        """Perform hyperparameter tuning for selected models"""
        if models_to_tune is None:
            # Focus on top performing models from baseline
            models_to_tune = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
        
        param_grids = self.get_hyperparameter_grids()
        tuned_results = {}
        
        for model_name in models_to_tune:
            if model_name not in self.models:
                continue
                
            logger.info(f"Tuning hyperparameters for {model_name}...")
            
            try:
                model = self.models[model_name]
                param_grid = param_grids.get(model_name, {})
                
                if search_method == 'grid':
                    search = GridSearchCV(
                        model, param_grid, cv=5, scoring='neg_mean_squared_error',
                        n_jobs=-1, verbose=1
                    )
                else:
                    search = RandomizedSearchCV(
                        model, param_grid, cv=5, scoring='neg_mean_squared_error',
                        n_jobs=-1, verbose=1, n_iter=50, random_state=self.random_state
                    )
                
                # Fit the search
                search.fit(X_train, y_train)
                
                # Get best model and predict
                best_model = search.best_estimator_
                y_pred = best_model.predict(X_val)
                
                # Calculate metrics
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                tuned_results[model_name] = {
                    'model': best_model,
                    'best_params': search.best_params_,
                    'best_score': search.best_score_,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                logger.info(f"{model_name} tuned - RMSE: {rmse:.2f}, R²: {r2:.4f}")
                logger.info(f"Best params: {search.best_params_}")
                
            except Exception as e:
                logger.error(f"Error tuning {model_name}: {str(e)}")
                tuned_results[model_name] = {'error': str(e)}
        
        self.results['tuned'] = tuned_results
        return tuned_results
    
    def ensemble_models(self, X_train, y_train, X_val, y_val, top_models=3):
        """Create ensemble of top performing models"""
        # Get top performing models from tuned results
        if 'tuned' in self.results:
            results = self.results['tuned']
        else:
            results = self.results.get('baseline', {})
        
        # Sort models by R² score
        sorted_models = sorted(
            [(name, res) for name, res in results.items() if 'r2' in res],
            key=lambda x: x[1]['r2'], reverse=True
        )
        
        top_model_list = []
        for name, res in sorted_models[:top_models]:
            top_model_list.append((name, res['model']))
        
        if len(top_model_list) < 2:
            logger.warning("Not enough models for ensemble")
            return {}
        
        # Create ensemble
        ensemble = VotingRegressor(top_model_list)
        ensemble.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = ensemble.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        ensemble_results = {
            'model': ensemble,
            'component_models': [name for name, _ in top_model_list],
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        self.results['ensemble'] = ensemble_results
        logger.info(f"Ensemble model - RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return ensemble_results
    
    def select_best_model(self):
        """Select the best performing model across all experiments"""
        all_results = {}
        
        # Collect all results
        for experiment_type, results in self.results.items():
            if experiment_type == 'ensemble':
                all_results['ensemble'] = results
            else:
                for model_name, metrics in results.items():
                    if 'r2' in metrics:
                        all_results[f"{experiment_type}_{model_name}"] = metrics
        
        # Find best model by R² score
        best_name = max(all_results.keys(), key=lambda x: all_results[x]['r2'])
        best_metrics = all_results[best_name]
        best_model = best_metrics['model']
        
        self.best_model = best_model
        self.best_params = best_metrics.get('best_params', {})
        
        logger.info(f"Best model selected: {best_name}")
        logger.info(f"Best R² score: {best_metrics['r2']:.4f}")
        
        return best_name, best_model, best_metrics
    
    def save_models(self, save_dir='models/trained_models'):
        """Save trained models and results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best model
        if self.best_model is not None:
            model_path = os.path.join(save_dir, 'best_model.joblib')
            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model saved to {model_path}")
        
        # Save all results
        results_path = os.path.join(save_dir, 'training_results.joblib')
        joblib.dump(self.results, results_path)
        logger.info(f"Training results saved to {results_path}")
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'random_state': self.random_state,
            'best_params': self.best_params
        }
        metadata_path = os.path.join(save_dir, 'training_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
    def training_pipeline(self, df, target_column='Salary'):
        """Complete training pipeline"""
        logger.info("Starting model training pipeline...")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df, target_column)
        
        # Train baseline models
        baseline_results = self.train_baseline_models(X_train, y_train, X_val, y_val)
        
        # Hyperparameter tuning for top models
        tuned_results = self.hyperparameter_tuning(X_train, y_train, X_val, y_val)
        
        # Create ensemble
        ensemble_results = self.ensemble_models(X_train, y_train, X_val, y_val)
        
        # Select best model
        best_name, best_model, best_metrics = self.select_best_model()
        
        # Save models
        self.save_models()
        
        logger.info("Model training pipeline completed successfully")
        
        return {
            'data_splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            },
            'best_model_info': {
                'name': best_name,
                'metrics': best_metrics
            },
            'all_results': self.results
        }

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    
    # Load engineered data
    df = pd.read_csv("data/processed/salary_data_engineered.csv")
    
    # Run training pipeline
    results = trainer.training_pipeline(df)
    
    print("Training completed!")
    print(f"Best model: {results['best_model_info']['name']}")
    print(f"Best R² score: {results['best_model_info']['metrics']['r2']:.4f}")