"""
Minimal Working Salary Prediction
Guaranteed to work without string conversion errors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

def minimal_salary_prediction():
    """Ultra-simple salary prediction that always works"""
    
    print("🎯 MINIMAL SALARY PREDICTION MODEL")
    print("=" * 40)
    
    # Step 1: Load data
    print("📊 Loading data...")
    df = pd.read_csv('Salary_Data_Based_country_and_race.csv')
    print(f"   Loaded {len(df)} records")
    
    # Step 2: Clean data
    print("🧹 Cleaning data...")
    
    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Remove rows with missing salary
    df = df.dropna(subset=['Salary'])
    
    # Fill missing values with mode/median
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    
    print(f"   Clean data: {len(df)} records")
    
    # Step 3: Encode categorical variables
    print("🔢 Encoding categorical variables...")
    
    categorical_columns = []
    label_encoders = {}
    
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Salary':
            categorical_columns.append(col)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"   ✓ Encoded {col}")
    
    # Step 4: Prepare features and target
    print("🎯 Preparing features...")
    
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"   Features: {X.shape[1]} columns")
    print(f"   Target range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    # Step 5: Split data
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 6: Train model
    print("🤖 Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully")
    
    # Step 7: Evaluate model
    print("📊 Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("\n🎯 MODEL PERFORMANCE:")
    print(f"   R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   MAE: ${mae:,.2f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # Step 8: Feature importance
    print("\n🔍 TOP 10 IMPORTANT FEATURES:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Step 9: Save results
    print("\n💾 Saving results...")
    
    # Create directories
    os.makedirs('models/trained_models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/trained_models/minimal_salary_model.joblib')
    print("   ✓ Model saved: models/trained_models/minimal_salary_model.joblib")
    
    # Save feature importance
    feature_importance.to_csv('reports/feature_importance_minimal.csv', index=False)
    print("   ✓ Feature importance saved: reports/feature_importance_minimal.csv")
    
    # Save performance metrics
    performance = {
        'model_type': 'RandomForestRegressor',
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': len(X.columns)
    }
    
    pd.DataFrame([performance]).to_csv('reports/model_performance_minimal.csv', index=False)
    print("   ✓ Performance metrics saved: reports/model_performance_minimal.csv")
    
    # Step 10: Create summary report
    print("\n📋 Creating summary report...")
    
    report = f"""
# 🎯 Minimal Salary Prediction Model - Results

## 📊 Model Performance
- **Algorithm**: Random Forest Regressor
- **R² Score**: {r2:.4f} ({r2*100:.1f}% variance explained)
- **RMSE**: ${rmse:,.2f}
- **MAE**: ${mae:,.2f}
- **MAPE**: {mape:.2f}%

## 📈 Dataset Summary
- **Total Records**: {len(df):,}
- **Features Used**: {len(X.columns)}
- **Training Samples**: {len(X_train):,}
- **Test Samples**: {len(X_test):,}
- **Salary Range**: ${y.min():,.0f} - ${y.max():,.0f}

## 🔍 Top 5 Important Features
{chr(10).join([f"{i+1}. {row['feature']}: {row['importance']:.4f}" for i, (_, row) in enumerate(feature_importance.head(5).iterrows())])}

## 💼 Business Impact
- **Use Case**: HR salary benchmarking and compensation planning
- **Accuracy**: Model explains {r2*100:.1f}% of salary variation
- **Error Range**: Typical prediction error of ${mae:,.0f}
- **Reliability**: {100-mape:.1f}% prediction accuracy

## 🚀 Next Steps
1. Use model for salary predictions: `joblib.load('models/trained_models/minimal_salary_model.joblib')`
2. Review feature importance for compensation insights
3. Collect additional data to improve model performance
4. Regular model retraining recommended (quarterly)

---
*Generated by Minimal Salary Prediction Pipeline*
"""
    
    with open('reports/MINIMAL_REPORT.md', 'w') as f:
        f.write(report)
    print("   ✓ Report saved: reports/MINIMAL_REPORT.md")
    
    print("\n🎉 SUCCESS! Minimal salary prediction completed!")
    print(f"📊 Model achieves {r2*100:.1f}% accuracy with ${mae:,.0f} average error")
    print("📁 Check reports/ folder for detailed results")
    
    return {
        'model': model,
        'performance': performance,
        'feature_importance': feature_importance,
        'encoders': label_encoders
    }

if __name__ == "__main__":
    try:
        results = minimal_salary_prediction()
        print("\n✅ All operations completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
