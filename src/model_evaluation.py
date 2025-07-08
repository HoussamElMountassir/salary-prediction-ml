"""
Model Evaluation Module for Salary Prediction Project

This module provides comprehensive model evaluation including
performance metrics, visualization, and bias analysis.

Author: [Your Name]
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve, validation_curve
import shap
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class for salary prediction
    """
    
    def __init__(self):
        self.results = {}
        self.model = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        
    def load_model_and_data(self, model_path, X_test, y_test):
        """
        Load trained model and test data
        
        Args:
            model_path (str): Path to saved model
            X_test: Test features
            y_test: Test targets
        """
        self.model = joblib.load(model_path)
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = self.model.predict(X_test)
        logger.info("Model and test data loaded successfully")
        
    def calculate_regression_metrics(self):
        """
        Calculate comprehensive regression metrics
        
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'mse': mean_squared_error(self.y_test, self.predictions),
            'rmse': np.sqrt(mean_squared_error(self.y_test, self.predictions)),
            'mae': mean_absolute_error(self.y_test, self.predictions),
            'r2': r2_score(self.y_test, self.predictions),
            'mape': mean_absolute_percentage_error(self.y_test, self.predictions) * 100,
            'explained_variance': 1 - (np.var(self.y_test - self.predictions) / np.var(self.y_test))
        }
        
        # Additional custom metrics
        residuals = self.y_test - self.predictions
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_error'] = np.max(np.abs(residuals))
        
        # Percentage of predictions within certain error bounds
        error_percentages = {}
        for threshold in [5000, 10000, 15000, 20000]:
            within_threshold = np.sum(np.abs(residuals) <= threshold) / len(residuals) * 100
            error_percentages[f'within_{threshold}'] = within_threshold
        
        metrics['error_percentages'] = error_percentages
        
        self.results['metrics'] = metrics
        logger.info("Regression metrics calculated")
        
        return metrics
    
    def plot_predictions_vs_actual(self, save_path=None):
        """
        Create predictions vs actual values plot
        
        Args:
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        axes[0].scatter(self.y_test, self.predictions, alpha=0.6, color='blue')
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Salary')
        axes[0].set_ylabel('Predicted Salary')
        axes[0].set_title('Predictions vs Actual Values')
        axes[0].grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(self.y_test, self.predictions)
        axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residual plot
        residuals = self.y_test - self.predictions
        axes[1].scatter(self.predictions, residuals, alpha=0.6, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Salary')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_error_distribution(self, save_path=None):
        """
        Plot error distribution analysis
        
        Args:
            save_path (str): Path to save the plot
        """
        residuals = self.y_test - self.predictions
        absolute_errors = np.abs(residuals)
        percentage_errors = (absolute_errors / self.y_test) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Residuals histogram
        axes[0, 0].hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Residuals')
        axes[0, 0].set_xlabel('Residuals')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(x=0, color='r', linestyle='--')
        
        # Absolute errors histogram
        axes[0, 1].hist(absolute_errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribution of Absolute Errors')
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        
        # Percentage errors histogram
        axes[1, 0].hist(percentage_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Distribution of Percentage Errors')
        axes[1, 0].set_xlabel('Percentage Error (%)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Q-Q plot for residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def feature_importance_analysis(self, feature_names=None, save_path=None):
        """
        Analyze and visualize feature importance
        
        Args:
            feature_names (list): Names of features
            save_path (str): Path to save the plot
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(self.X_test.shape[1])]
            
        # Try to get feature importance from the model
        importance = None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        
        if importance is not None:
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title('Top 20 Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            plt.show()
            
            return importance_df
        else:
            logger.warning("Model does not have feature importance attribute")
            return None
    
    def shap_analysis(self, sample_size=100, save_path=None):
        """
        SHAP (SHapley Additive exPlanations) analysis
        
        Args:
            sample_size (int): Number of samples for SHAP analysis
            save_path (str): Path to save plots
        """
        try:
            # Sample data for SHAP analysis (computationally expensive)
            if len(self.X_test) > sample_size:
                sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
                X_sample = self.X_test.iloc[sample_indices] if hasattr(self.X_test, 'iloc') else self.X_test[sample_indices]
            else:
                X_sample = self.X_test
            
            # Create SHAP explainer
            explainer = shap.Explainer(self.model, X_sample)
            shap_values = explainer(X_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title('SHAP Summary Plot')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_summary.png", dpi=300, bbox_inches='tight')
                
            plt.show()
            
            # Feature importance plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_importance.png", dpi=300, bbox_inches='tight')
                
            plt.show()
            
            return shap_values
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {str(e)}")
            return None
    
    def bias_analysis(self, demographic_data, save_path=None):
        """
        Analyze model bias across demographic groups
        
        Args:
            demographic_data (pd.DataFrame): Demographic information
            save_path (str): Path to save plots
        """
        results = {}
        
        # Combine predictions with demographic data
        analysis_df = demographic_data.copy()
        analysis_df['Actual'] = self.y_test
        analysis_df['Predicted'] = self.predictions
        analysis_df['Error'] = self.y_test - self.predictions
        analysis_df['Abs_Error'] = np.abs(analysis_df['Error'])
        analysis_df['Percentage_Error'] = (analysis_df['Abs_Error'] / analysis_df['Actual']) * 100
        
        # Analyze bias by demographic groups
        demographic_cols = ['Gender', 'Race', 'Country', 'Education Level']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for i, col in enumerate(demographic_cols):
            if col not in analysis_df.columns:
                continue
                
            row, col_idx = divmod(i, 2)
            
            # Calculate metrics by group
            group_metrics = analysis_df.groupby(col).agg({
                'Error': ['mean', 'std'],
                'Abs_Error': 'mean',
                'Percentage_Error': 'mean'
            }).round(3)
            
            results[col] = group_metrics
            
            # Plot mean absolute error by group
            mae_by_group = analysis_df.groupby(col)['Abs_Error'].mean()
            mae_by_group.plot(kind='bar', ax=axes[row, col_idx], color='skyblue')
            axes[row, col_idx].set_title(f'Mean Absolute Error by {col}')
            axes[row, col_idx].set_ylabel('Mean Absolute Error')
            axes[row, col_idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
        # Print statistical summary
        print("🔍 BIAS ANALYSIS RESULTS:")
        print("=" * 50)
        
        for col, metrics in results.items():
            print(f"\n--- {col} ---")
            print(metrics)
            
        return results
    
    def learning_curve_analysis(self, train_sizes=None, save_path=None):
        """
        Generate and plot learning curves
        
        Args:
            train_sizes (array): Training set sizes to evaluate
            save_path (str): Path to save the plot
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
            
        # Combine training and test data for learning curve
        X_combined = np.vstack([self.X_test, self.X_test])  # This would need actual training data
        y_combined = np.hstack([self.y_test, self.y_test])  # This would need actual training data
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.model, X_combined, y_combined, 
                train_sizes=train_sizes, cv=5, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            # Convert to positive RMSE
            train_rmse = np.sqrt(-train_scores)
            val_rmse = np.sqrt(-val_scores)
            
            plt.figure(figsize=(12, 8))
            
            plt.plot(train_sizes_abs, np.mean(train_rmse, axis=1), 
                    'o-', color='blue', label='Training RMSE')
            plt.fill_between(train_sizes_abs, 
                           np.mean(train_rmse, axis=1) - np.std(train_rmse, axis=1),
                           np.mean(train_rmse, axis=1) + np.std(train_rmse, axis=1),
                           alpha=0.2, color='blue')
            
            plt.plot(train_sizes_abs, np.mean(val_rmse, axis=1), 
                    'o-', color='red', label='Validation RMSE')
            plt.fill_between(train_sizes_abs,
                           np.mean(val_rmse, axis=1) - np.std(val_rmse, axis=1),
                           np.mean(val_rmse, axis=1) + np.std(val_rmse, axis=1),
                           alpha=0.2, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('RMSE')
            plt.title('Learning Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            plt.show()
            
        except Exception as e:
            logger.error(f"Learning curve analysis failed: {str(e)}")
    
    def prediction_intervals(self, confidence_level=0.95):
        """
        Calculate prediction intervals
        
        Args:
            confidence_level (float): Confidence level for intervals
            
        Returns:
            tuple: Lower and upper prediction intervals
        """
        residuals = self.y_test - self.predictions
        residual_std = np.std(residuals)
        
        # Calculate confidence interval
        from scipy import stats
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha/2, len(residuals) - 1)
        
        margin_of_error = t_value * residual_std
        
        lower_bound = self.predictions - margin_of_error
        upper_bound = self.predictions + margin_of_error
        
        # Calculate coverage
        coverage = np.mean((self.y_test >= lower_bound) & (self.y_test <= upper_bound))
        
        print(f"Prediction Interval Coverage: {coverage:.3f}")
        print(f"Expected Coverage: {confidence_level:.3f}")
        
        return lower_bound, upper_bound
    
    def comprehensive_evaluation_report(self, demographic_data=None, save_dir='reports'):
        """
        Generate comprehensive evaluation report
        
        Args:
            demographic_data (pd.DataFrame): Demographic information for bias analysis
            save_dir (str): Directory to save reports
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("📊 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" * 60)
        
        # Calculate metrics
        metrics = self.calculate_regression_metrics()
        
        print("\n🎯 PERFORMANCE METRICS:")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: ${metrics['rmse']:,.2f}")
        print(f"MAE: ${metrics['mae']:,.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        print("\n📈 ERROR ANALYSIS:")
        for threshold, percentage in metrics['error_percentages'].items():
            print(f"Predictions within ${threshold:,}: {percentage:.1f}%")
        
        # Generate plots
        self.plot_predictions_vs_actual(f"{save_dir}/predictions_vs_actual.png")
        self.plot_error_distribution(f"{save_dir}/error_distribution.png")
        
        # Feature importance
        importance_df = self.feature_importance_analysis(save_path=f"{save_dir}/feature_importance.png")
        
        # SHAP analysis
        shap_values = self.shap_analysis(save_path=f"{save_dir}/shap_analysis")
        
        # Bias analysis
        if demographic_data is not None:
            bias_results = self.bias_analysis(demographic_data, f"{save_dir}/bias_analysis.png")
        
        # Prediction intervals
        lower_bound, upper_bound = self.prediction_intervals()
        
        print("\n✅ EVALUATION COMPLETE!")
        print(f"Reports saved to: {save_dir}/")
        
        return {
            'metrics': metrics,
            'importance': importance_df,
            'shap_values': shap_values,
            'bias_results': bias_results if demographic_data is not None else None
        }

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Load model and data (example)
    # evaluator.load_model_and_data('models/best_model.joblib', X_test, y_test)
    
    # Generate comprehensive report
    # results = evaluator.comprehensive_evaluation_report(demographic_data)
    
    print("Model evaluation module ready!")