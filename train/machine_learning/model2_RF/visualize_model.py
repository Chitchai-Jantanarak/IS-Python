import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.inspection import permutation_importance

import sys
sys.path.append('..')
from EDA import data_cleasing

from builder_model import build_rf_model
from crossval import evaluate_model

# Caller function
def visualize(model=None, X=None, y=None, X_test=None, y_test=None, y_pred=None):
    """
    Generate visualizations for the Random Forest model
    Args:
        model: Trained model (optional)
        X, y: Full dataset (optional)
        X_test, y_test: Test data (optional)
        y_pred: Model predictions (optional)
    """
    # Load the model and data if not provided
    if model is None or X is None or y is None or X_test is None or y_test is None or y_pred is None:
        
        model, X_train, X_test, y_train, y_test, X, y = build_rf_model()
        y_pred, _, _, _, _ = evaluate_model(model, X_test, X_train, y_test, y_train)
    
    # Load feature data
    feature_data = joblib.load('models/feature_data.pkl')
    numerical_feat = feature_data['numerical_feat']
    categorical_feat = feature_data['categorical_feat']
    
    # Create a results DataFrame for visualizations
    df = data_cleansing()
    test_indices = X_test.index
    
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Absolute_Error': np.abs(y_test - y_pred),
        'Species': df.loc[test_indices, 'Species'],
        'Flipper_Length': df.loc[test_indices, 'Flipper Length (mm)'],
        'Culmen_Length': df.loc[test_indices, 'Culmen Length (mm)'],
        'Culmen_Depth': df.loc[test_indices, 'Culmen Depth (mm)'],
        'Sex': df.loc[test_indices, 'Sex']
    })
    
    # Generate all visualizations
    importance_feat(model, numerical_feat, categorical_feat)
    prediction_plot(y_test, y_pred)
    residual_plot(y_test, y_pred)
    morphological_feat(results_df)
    distribution_plot(results_df)
    
    print("All visualizations have been generated in the 'visualizations' directory.")

def importance_feat(model, numerical_feat, categorical_feat):
    """Generate feature importance plot"""
    rf_model = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']

    # Get feature names after one-hot encoding
    categorical_encoder = preprocessor.transformers_[1][1]
    cat_features = categorical_encoder.get_feature_names_out(categorical_feat)
    
    # Combine all feature names
    feature_names = np.array(numerical_feat + list(cat_features))
    
    # Get feature importances
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    top_features = importances.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top Features for Predicting Penguin Body Mass', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300)
    plt.close()

def prediction_plot(y_test, y_pred):
    """Generate actual vs predicted plot"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Body Mass (g)', fontsize=12)
    plt.ylabel('Predicted Body Mass (g)', fontsize=12)
    plt.title('Actual vs. Predicted Body Mass', fontsize=14)
    plt.savefig('visualizations/actual_vs_predicted.png', dpi=300)
    plt.close()

def residual_plot(y_test, y_pred):
    """Generate residual plot"""
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolor='k')
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--', lw=2)
    plt.xlabel('Predicted Body Mass (g)', fontsize=12)
    plt.ylabel('Residuals (g)', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.savefig('visualizations/residuals.png', dpi=300)
    plt.close()

def morphological_feat(results_df):
    """Generate morphological features plot"""
    plt.figure(figsize=(12, 10))
    accuracy_colors = plt.cm.viridis(results_df['Absolute_Error'] / results_df['Absolute_Error'].max())

    plt.scatter(
        results_df['Flipper_Length'], 
        results_df['Culmen_Length'],
        c=accuracy_colors, 
        alpha=0.7,
        s=100, 
        edgecolor='k'
    )

    cbar = plt.colorbar()
    cbar.set_label('Prediction Error (g)', fontsize=12)
    plt.xlabel('Flipper Length (mm)', fontsize=12)
    plt.ylabel('Culmen Length (mm)', fontsize=12)
    plt.title('Morphological Features and Prediction Accuracy', fontsize=14)
    plt.savefig('visualizations/morphology_vs_accuracy.png', dpi=300)
    plt.close()

def distribution_plot(results_df):
    """Generate distribution plot by species"""
    plt.figure(figsize=(12, 8))
    for species in results_df['Species'].unique():
        species_data = results_df[results_df['Species'] == species]
        sns.kdeplot(species_data['Predicted'], label=species, shade=True)
    plt.xlabel('Predicted Body Mass (g)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Predicted Body Mass by Species', fontsize=14)
    plt.legend()
    plt.savefig('visualizations/prediction_distribution.png', dpi=300)
    plt.close()

# main.py
def main():
    """Main script to run the entire pipeline"""
    print("Starting penguin data analysis pipeline...")
    
    # 1. Data preparation
    from EDA import data_cleansing
    print("Cleaning data...")
    data_cleansing()
    
    # 2. Build model
    from builder_model import build_rf_model
    print("Building Random Forest regression model...")
    model, X_train, X_test, y_train, y_test, X, y = build_rf_model()
    
    # 3. Evaluate model
    from evaluate_model import evaluate_model
    print("Evaluating model performance...")
    y_pred, rmse, mae, r2, cv_scores = evaluate_model(model, X_test, X_train, y_test, y_train)
    
    # 4. Visualize results
    from visualize_model import visualize
    print("Generating visualizations...")
    visualize(model, X, y, X_test, y_test, y_pred)
    
    print("\nComplete! Check the 'visualizations' directory for results.")
