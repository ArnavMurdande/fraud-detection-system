import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# Set plotting style
plt.style.use('ggplot')

MODEL_PATH = "models/xgb_fraud_model.json"
TEST_DATA_PATH = "data/test_set.csv"
RESULTS_DIR = "results/"

def explain_model():
    print("Loading model and test data...")
    # Load model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Load test data
    df = pd.read_csv(TEST_DATA_PATH)
    X_test = df.drop(columns=['is_fraud'])
    y_test = df['is_fraud']
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --- SHAP Calculation ---
    print("Computing SHAP values...")
    # Using TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Limit to 500 samples for performance
    sample_size = min(500, len(X_test))
    shap_values = explainer.shap_values(X_test.iloc[:sample_size])
    
    # --- 1. Global Feature Importance (Summary Plot) ---
    print("Generating Global Feature Importance Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test.iloc[:sample_size], show=False)
    
    summary_plot_path = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(summary_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # --- Top Features Calculation ---
    # Convert shap_values to meaningful importance
    # Mean absolute SHAP value per feature
    if isinstance(shap_values, list): # For some older SHAP versions/binary logistic return list
        vals = np.abs(shap_values[1]).mean(0) # Index 1 for positive class
    else:
        vals = np.abs(shap_values).mean(0)
        
    feature_importance = pd.DataFrame(list(zip(X_test.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    top_3_features = feature_importance['col_name'].head(3).tolist()
    
    print("-" * 30)
    print("Top 3 Predictive Features (SHAP):")
    for i, f in enumerate(top_3_features, 1):
        print(f"{i}. {f}")
    print("-" * 30)
    
    # --- 2. Local Explanation (Single Fraud Case) ---
    print("Generating Local Explanation for a correctly detected fraud case...")
    y_pred = model.predict(X_test)
    
    # Find index of a True Positive
    # We iterate through the whole test set index, but remember SHAP was computed only on first 500
    # Let's find a TP within the first 500 if possible, or computing shap for single instance if needed.
    # To use existing shap_values, we need index < sample_size.
    
    tp_indices = np.where((y_test.iloc[:sample_size] == 1) & (y_pred[:sample_size] == 1))[0]
    
    if len(tp_indices) > 0:
        idx = tp_indices[0] # Take first TP
        print(f" explaining transaction at index {idx}...")
        
        # Force plot
        # force_plot returns a visualisation object. save_html requires writing it.
        # Initialize JS for HTML output
        shap.initjs()
        
        plot = shap.force_plot(
            explainer.expected_value, 
            shap_values[idx], 
            X_test.iloc[idx],
            matplotlib=False,
            show=False
        )
        
        force_plot_path = os.path.join(RESULTS_DIR, "shap_fraud_explanation.html")
        shap.save_html(force_plot_path, plot)
    else:
        print("No True Positive found in the first 500 samples. Skipping Force Plot.")
        force_plot_path = "Skipped"

    # --- 3. Feature Dependence Plot ---
    # Using the #1 most important feature
    top_feature = top_3_features[0]
    print(f"Generating Dependence Plot for top feature: {top_feature}...")
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_feature, 
        shap_values, 
        X_test.iloc[:sample_size],
        show=False
    )
    
    dependence_plot_path = os.path.join(RESULTS_DIR, "shap_dependence.png")
    # dependence_plot uses matplotlib directly if interaction_index is not 'auto' or matplotlib=True
    # By default it plots. We need to save the current figure.
    plt.savefig(dependence_plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    # --- Final Output ---
    print("\n" + "="*40)
    print("EXPLAINABILITY REPORT GENERATED")
    print("="*40)
    print(f"Summary Plot:     {summary_plot_path}")
    print(f"Dependence Plot:  {dependence_plot_path}")
    print(f"Local Fraud Expl: {force_plot_path}")
    print("-" * 40)
    print("CONFIRMATION: Explanations generated on UNSEEN TEST DATA.")
    print("SHAP provides transparent, local explanations for every prediction,")
    print("building trust in the AI's fraud detection capabilities.")
    print("="*40)

if __name__ == "__main__":
    explain_model()
