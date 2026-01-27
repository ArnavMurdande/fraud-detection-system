import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

INPUT_PATH = "data/train_data.csv"
MODEL_PATH = "models/xgb_fraud_model.json"
TEST_SET_OUTPUT = "data/test_set.csv"

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    # Ensure no NaNs - XGBoost handles them but for clarity let's fill
    # Actually, previous step handled most. Let's just trust X and y.
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    return X, y

def train_model():
    X, y = load_data(INPUT_PATH)
    
    # --- Time-Aware Splitting ---
    # Data is already time sorted.
    # Split 80/20 train/test
    # Then split recent part of train for validation
    
    total_rows = len(X)
    train_size = int(total_rows * 0.8)
    
    X_train_full = X.iloc[:train_size]
    y_train_full = y.iloc[:train_size]
    
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    # Further split training for validation (last 10% of training data)
    # This simulates "recency" validation
    val_size = int(len(X_train_full) * 0.1)
    train_split_idx = len(X_train_full) - val_size
    
    X_train = X_train_full.iloc[:train_split_idx]
    y_train = y_train_full.iloc[:train_split_idx]
    
    X_val = X_train_full.iloc[train_split_idx:]
    y_val = y_train_full.iloc[train_split_idx:]
    
    print("-" * 30)
    print(f"Train size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")
    print(f"Test size:  {len(X_test)}")
    print("-" * 30)
    
    # --- Class Imbalance Handling ---
    # scale_pos_weight = neg / pos
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"Class Imbalance: {pos_count} Fraud vs {neg_count} Normal")
    print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")
    
    # --- Model Configuration ---
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    print("Training XGBoost with early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"Best Iteration: {model.best_iteration}")
    
    # --- Evaluation ---
    print("Evaluating on Test Set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Why Recall is prioritized:
    # In fraud detection, a False Negative (missing a fraud) is much more costly
    # than a False Positive (alerting a customer). We want high Recall to catch
    # as many fraudulent transactions as possible, even if precision drops slightly.
    
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)
    
    print("\n" + "="*40)
    print("CONFUSION MATRIX")
    print("="*40)
    print(cm)
    print("\n" + "="*40)
    print("CLASSIFICATION REPORT")
    print("="*40)
    print(report)
    print(f"ROC-AUC Score: {roc:.4f}")

    # --- Plot Confusion Matrix ---
    print("\nGenerating Confusion Matrix Plot...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix')
    
    cm_path = "results/confusion_matrix.png"
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    print(f"Confusion Matrix saved to {cm_path}")
    plt.show()
    
    # --- Artifacts ---
    print("\nSaving artifacts...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save test set for explainability
    test_df = X_test.copy()
    test_df['is_fraud'] = y_test
    test_df.to_csv(TEST_SET_OUTPUT, index=False)
    print(f"Test set saved to {TEST_SET_OUTPUT}")

if __name__ == "__main__":
    train_model()
