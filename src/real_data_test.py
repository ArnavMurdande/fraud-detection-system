import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime, timedelta

# Import feature engineering logic
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from feature_engineering import (
        create_temporal_features,
        create_velocity_features,
        create_spending_behavior_features,
        create_history_based_features,
        create_interaction_features
    )
except ImportError:
    print("Error imports from src/feature_engineering.py. Ensure the file exists.")
    sys.exit(1)

# Options
pd.set_option('display.max_columns', None)

# Paths
REAL_DATA_DIR = "data/real_data"
MODEL_PATH = "models/xgb_fraud_model.json"
RESULTS_DIR = "results/real_data_eval"
SYNTHETIC_DATA_PATH = "data/transactions.csv" # To reuse encoders logic properly

def load_real_data():
    """
    Attempts to load BankSim or PaySim from data/real_data.
    Prioritizes BankSim if both exist for this exercise.
    """
    if not os.path.exists(REAL_DATA_DIR):
        print(f"Directory {REAL_DATA_DIR} not found. Please create it and add data.")
        return None, None

    files = os.listdir(REAL_DATA_DIR)
    
    # 1. Check for PaySim ARFF/CSV
    paysim_arff = next((f for f in files if 'paysim' in f.lower() and f.endswith('.arff')), None)
    paysim_csv = next((f for f in files if 'paysim' in f.lower() and f.endswith('.csv')), None)
    
    # 2. Check for BankSim
    banksim_file = next((f for f in files if 'banksim' in f.lower() and 'csv' in f), None)
    
    df = None
    dataset_name = "Unknown"
    
    # Logic: Load PaySim if found (as per prompt mentioning it), else BankSim
    if paysim_arff or paysim_csv:
        dataset_name = "PaySim"
        path = os.path.join(REAL_DATA_DIR, paysim_csv if paysim_csv else paysim_arff)
        
        if path.endswith('.arff'):
            print(f"Converting ARFF: {path}...")
            data, meta = arff.loadarff(path)
            df = pd.DataFrame(data)
            # Save as CSV for future
            csv_path = path.replace('.arff', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved converted CSV to {csv_path}")
        else:
            print(f"Loading PaySim CSV: {path}...")
            df = pd.read_csv(path)
            
    elif banksim_file:
        dataset_name = "BankSim"
        path = os.path.join(REAL_DATA_DIR, banksim_file)
        print(f"Loading BankSim CSV: {path}...")
        df = pd.read_csv(path)
        
    else:
        print("No supported data found in data/real_data. Please add BankSim or PaySim files.")
        return None, None

    return df, dataset_name

def align_features(df, dataset_name):
    """
    Maps real data columns to the synthetic training schema.
    """
    print(f"Aligning features for {dataset_name}...")
    
    aligned_df = pd.DataFrame()
    
    # Synthetic Schema targets
    # user_id, amount, timestamp (derived), category, payment_method, device_id, location, age, gender
    
    # Base timestamp for step conversion (Arbitrary start)
    start_date = datetime(2025, 1, 1)
    
    if dataset_name == "PaySim":
        # PaySim Cols: step, type, amount, nameOrig, oldbalanceOrg... isFraud
        
        # Sampling if large
        if len(df) > 500000:
            print("Dataset too large. Performing stratified sampling (max 200k rows)...")
            # Stratified sample
            fraction = 200000 / len(df)
            # Check for label column safely
            label_col = 'isFraud' if 'isFraud' in df.columns else ('fraud' if 'fraud' in df.columns else None)
            
            if label_col:
                df = df.groupby(label_col, group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=42))
            else:
                # Fallback to random sampling if no label found (unlikely given logic above evaluates availability)
                print("Warning: Label column not found for stratification. Random sampling.")
                df = df.sample(frac=fraction, random_state=42)
            df = df.sort_values('step') # Restore order
            
        # Clean columns just in case
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
        print(f"PaySim Columns: {list(df.columns)}")

        aligned_df['amount'] = df['amount']
        aligned_df['user_id'] = df['nameOrig']
        aligned_df['merchant_id'] = df['nameDest']
        # Map step (hours) to timestamp
        # vectorized date addition
        aligned_df['timestamp'] = df['step'].apply(lambda x: start_date + timedelta(hours=int(x)))
        
        # Category Mapping (PaySim 'type' is closer to method, but let's use it for category/method)
        # Type values: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
        # Map to synthetic categories roughly
        cat_map = {
            'PAYMENT': 'es_otherservices',
            'TRANSFER': 'es_transportation', # Proxy
            'CASH_OUT': 'es_wellnessandbeauty', # Random mapping for alignment
            'DEBIT': 'es_health',
            'CASH_IN': 'es_tech'
        }
        aligned_df['category'] = df['type'].map(cat_map).fillna('es_otherservices')
        aligned_df['payment_method'] = 'Digital Wallet' # Default
        
        if 'isFraud' in df.columns:
            aligned_df['is_fraud'] = df['isFraud']
        else:
             # Try case insensitive search
             possible_col = next((c for c in df.columns if c.lower() == 'isfraud'), None)
             if possible_col:
                 aligned_df['is_fraud'] = df[possible_col]
             else:
                 print("CRITICAL WARNING: 'isFraud' column not found. Cannot evaluate accuracy.")
                 aligned_df['is_fraud'] = 0 # Dummy to avoid crash, but metrics will be wrong
        
        # Missing Coils defaults
        aligned_df['device_id'] = 'unknown_device'
        aligned_df['location'] = 'US'
        aligned_df['age'] = 30
        aligned_df['gender'] = 'M'
        
    elif dataset_name == "BankSim":
        # BankSim Cols match closer usually: step, customer, age, gender, merchant... fraud
        
        if len(df) > 500000:
             # Stratified sample
            fraction = 200000 / len(df)
            df = df.groupby('fraud', group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=42))
            df = df.sort_values('step')
            
        # Map columns
        col_map = {
            'amount': 'amount',
            'customer': 'user_id',
            'merchant': 'merchant_id',
            'category': 'category'
        }
        # Check if cols exist, BankSim columns usually quoted like 'amount'
        # Clean quotes if strictly raw csv
        df.columns = df.columns.str.replace("'", "").str.strip()
        
        for tgt, src in col_map.items():
            if src in df.columns:
                aligned_df[tgt] = df[src]
                
        # Timestamp
        if 'step' in df.columns:
            # BankSim step is usually days
            aligned_df['timestamp'] = df['step'].apply(lambda x: start_date + timedelta(days=int(x)))
        else:
            print("Warning: No step column. creating dummy time.")
            aligned_df['timestamp'] = [start_date + timedelta(minutes=i) for i in range(len(df))]
            
        aligned_df['is_fraud'] = df['fraud'] if 'fraud' in df.columns else 0
        
        # Defaults
        aligned_df['payment_method'] = 'Credit Card'
        aligned_df['device_id'] = 'unknown_device'
        aligned_df['location'] = 'US'
        aligned_df['age'] = df['age'] if 'age' in df.columns else 40
        aligned_df['gender'] = df['gender'] if 'gender' in df.columns else 'F'
        
        # Handle BankSim categories usually like 'es_transportation' matches synthetic
        # If string format differs, might need cleaning
        aligned_df['category'] = aligned_df['category'].astype(str).str.replace("'", "")

    return aligned_df

def fit_encoders(synthetic_path):
    """
    Fits LabelEncoders on the ORIGINAL synthetic data to ensure consistent mapping.
    """
    print("Fitting encoders on original synthetic data...")
    if not os.path.exists(synthetic_path):
        print("Warning: Synthetic data not found. Encoding might be unstable.")
        return {}
        
    df = pd.read_csv(synthetic_path)
    encoders = {}
    cols = ['category', 'payment_method', 'gender', 'location']
    
    # Also need combination combo encoder
    df['method_category_combo'] = df['payment_method'].astype(str) + "_" + df['category'].astype(str)
    cols.append('method_category_combo')
    
    for col in cols:
        le = LabelEncoder()
        # Ensure string
        vals = df[col].astype(str).unique()
        # Add 'unknown' handle just in case for real data
        vals = np.append(vals, 'unknown') 
        le.fit(vals)
        encoders[col] = le
        
    return encoders

def transform_with_encoders(df, encoders):
    """
    Applies fitted encoders. Handles unseen labels by mapping to 'unknown' or mode.
    """
    print("Transforming categorical features...")
    cols = ['category', 'payment_method', 'gender', 'method_category_combo', 'location']
    
    for col in cols:
        if col not in encoders:
            # Skip if we couldn't fit
            continue
            
        le = encoders[col]
        # Get data classes
        known_classes = set(le.classes_)
        
        # Replace unseen with known class (e.g. first one) or specific unknown if added
        # We added 'unknown' to classes in fit_encoders
        fallback = 'unknown' if 'unknown' in known_classes else le.classes_[0]
        
        df[col] = df[col].astype(str).apply(lambda x: x if x in known_classes else fallback)
        df[col] = le.transform(df[col])
        
    return df

def evaluate_on_real_data():
    # 1. Load Real Data
    real_df, name = load_real_data()
    if real_df is None:
        return

    # 2. Align Data
    df = align_features(real_df, name)
    
    # 3. Feature Engineering (Reuse logic)
    print("Applying feature engineering variables...")
    # Add weekends
    df = create_temporal_features(df)
    
    # Velocity (needs time sorting)
    df.sort_values('timestamp', inplace=True)
    df = create_velocity_features(df)
    
    # Spending
    df = create_spending_behavior_features(df)
    
    # History
    df = create_history_based_features(df)
    
    # Interaction
    df = create_interaction_features(df)
    
    # 4. Encoding
    encoders = fit_encoders(SYNTHETIC_DATA_PATH)
    df = transform_with_encoders(df, encoders)
    
    # Cleanup for inference
    target = df['is_fraud']
    features = df.drop(columns=['is_fraud', 'transaction_id', 'user_id', 'merchant_id', 'device_id', 'timestamp'], errors='ignore')
    
    # Ensure columns match model expectation
    # Model expects specific column order/names.
    # Load model to check feature names? XGBoost saves feature names.
    
    # 5. Model Inference
    print("Loading model for inference...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Get feature names from training data artifacts to ensure correctness
    print("Loading feature names from training configuration...")
    train_cols_df = pd.read_csv('data/train_data.csv', nrows=0)
    model_features = [c for c in train_cols_df.columns if c != 'is_fraud']
    print(f"Model expects {len(model_features)} features: {model_features}")
    
    # Add missing cols with 0 if any (robustness)
    for f in model_features:
        if f not in features.columns:
            features[f] = 0
            
    # Select only model features in order
    features = features[model_features]
    
    print("Predicting on real data...")
    y_pred = model.predict(features)
    y_prob = model.predict_proba(features)[:, 1]
    
    # 6. Evaluation
    print("Computing metrics...")
    cm = confusion_matrix(target, y_pred)
    report = classification_report(target, y_pred)
    roc = roc_auc_score(target, y_prob)
    
    # 7. Output
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("\n" + "="*40)
    print(f"EVALUATION REPORT: {name}")
    print("="*40)
    print(f"Evaluated Rows: {len(df)}")
    print(f"Fraud Rate:     {(target.mean()*100):.2f}%")
    print("-" * 40)
    print(f"ROC-AUC: {roc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("="*40)
    
    # Save Metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics_real.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"ROC-AUC: {roc}\n")
        f.write(f"Report:\n{report}\n")
        
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title(f'Confusion Matrix ({name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_real.png"))
    plt.close()
    
    # Save Predictions
    res_df = pd.DataFrame({
        'true_label': target,
        'predicted_label': y_pred,
        'probability': y_prob
    })
    res_df.to_csv(os.path.join(RESULTS_DIR, "predictions_real.csv"), index=False)
    print(f"Results saved to {RESULTS_DIR}")
    
    # Comments on Domain Shift
    print("\n[NOTE] Domain Shift Analysis:")
    print(" Performance on real data may vary significantly from synthetic results.")
    print(" Factors: Feature distribution mismatch, different fraud patterns in PaySim/BankSim,")
    print(" and encoding differences (e.g., 'es_transportation' ID might differ if not perfectly mapped).")
    print(" This script attempts best-effort alignment using synthetic encoders.")

if __name__ == "__main__":
    evaluate_on_real_data()
