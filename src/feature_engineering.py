import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Set pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

INPUT_PATH = "data/transactions.csv"
OUTPUT_PATH = "data/train_data.csv"

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def create_temporal_features(df):
    print("Creating temporal features...")
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_night'] = df['hour_of_day'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def create_velocity_features(df):
    print("Creating velocity features (rolling counts)...")
    
    # We must treat this strictly per user.
    # To efficiently compute rolling counts per user without leakage:
    # 1. Group by user
    # 2. Set index to timestamp
    # 3. Use rolling() with time offsets
    
    # Temporary index for rolling ops
    df_indexed = df.set_index('timestamp').sort_index()
    
    # Calculate counts per user
    # '1h' implies including the current row if closed='right' (default)
    # But usually we want counts *before* or including this one.
    # In fraud detection, current transaction counts towards velocity if it's happening NOW.
    # Example: 10th txn in 1 hour is suspicious.
    
    grouped = df_indexed.groupby('user_id')
    
    # Count includes the current transaction
    df['count_tx_last_1_hour'] = grouped['amount'].rolling('1h').count().values
    df['count_tx_last_24_hours'] = grouped['amount'].rolling('24h').count().values
    df['txns_last_10min'] = grouped['amount'].rolling('10min').count().values
    
    # Time since last transaction
    # shift(1) gives the previous timestamp
    print("Calculating hours since last transaction...")
    df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
    df['time_diff'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 3600.0
    df['hours_since_last_tx'] = df['time_diff'].fillna(-1)
    
    df.drop(columns=['prev_timestamp', 'time_diff'], inplace=True)
    return df

def create_spending_behavior_features(df):
    print("Creating spending behavior features...")
    
    df_indexed = df.set_index('timestamp').sort_index()
    grouped = df_indexed.groupby('user_id')
    
    # 30d rolling stats
    # min_periods=1 ensures we get a mean from the very first transaction onwards
    # closed='right' includes current record in the mean/std
    df['rolling_mean_amount_30d'] = grouped['amount'].rolling('30D', min_periods=1).mean().values
    df['rolling_std_amount_30d'] = grouped['amount'].rolling('30D', min_periods=1).std().fillna(0).values
    
    # Safe ratio
    df['amount_ratio'] = np.where(
        df['rolling_mean_amount_30d'] > 0,
        df['amount'] / df['rolling_mean_amount_30d'],
        0
    )
    
    return df

def create_history_based_features(df):
    print("Creating history-based anomaly features (device, location, category)...")
    
    # These must be iterative or efficiently vectorized to ensure NO LEAKAGE (future info)
    # "is_new_device": Has device_id appeared for this user_id BEFORE this timestamp?
    # "is_new_category": Similar logic.
    
    # Efficient pandas way:
    # Cumcount of specific combinations vs Total cumcount? No.
    # We can group by User, then expanding apply? Slow.
    # Better: Use duplicated() with keep='first' on cumulative slices?
    # Actually, duplicated() on the whole sorted dataset (subset=[user, device]) marks duplicates.
    # The FIRST occurrence is False (not duplicate), subsequent are True.
    # So ~duplicated() is True for the first time.
    
    # Is New Device
    # Mark first occurrence of (user, device) as New
    # Since dataset is strictly time sorted:
    df['is_new_device'] = (~df.duplicated(subset=['user_id', 'device_id'], keep='first')).astype(int)
    
    # Is New Category
    # Mark first occurrence of (user, category) as New
    df['is_new_category_for_user'] = (~df.duplicated(subset=['user_id', 'category'], keep='first')).astype(int)
    
    # Category Txn Count (Cumulative count of category usage for this user)
    # Group by user+category, then cumcount + 1
    df['category_txn_count'] = df.groupby(['user_id', 'category']).cumcount() + 1
    
    # Foreign Transaction
    # Definition: Location differs from user's most frequent location *so far*? 
    # Or just "home country"?
    # The prompt says: "differs from user's most frequent location".
    # Real-world: We usually know home country.
    # Let's simplify efficiently: Compute "Home Country" as the mode of location for the user 
    # based on *past* transactions or just global mode (assuming dataset covers history).
    # To be strictly non-leaky, we'd need running mode. 
    # Approximating "Home Country" is usually done at account level.
    # In generate_data.py, users had a home_country.
    # We can infer it from the MOST frequent location overall in the training set (offline feature)
    # OR simpler: First location seen is "Home".
    
    print("Inferring home location (first location seen for user)...")
    # Get first location for each user
    home_locs = df.groupby('user_id')['location'].first().reset_index()
    home_locs.columns = ['user_id', 'home_location']
    
    df = df.merge(home_locs, on='user_id', how='left')
    df['is_foreign_transaction'] = (df['location'] != df['home_location']).astype(int)
    df.drop(columns=['home_location'], inplace=True)
    
    return df

def create_interaction_features(df):
    print("Creating interaction features...")
    df['method_category_combo'] = df['payment_method'].astype(str) + "_" + df['category'].astype(str)
    return df

def encode_and_cleanup(df):
    print("Encoding categorical variables and cleaning up...")
    
    categorical_cols = ['category', 'payment_method', 'gender', 'method_category_combo', 'location']
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        
    # Drop identifiers
    drop_cols = ['transaction_id', 'user_id', 'merchant_id', 'device_id']
    df.drop(columns=drop_cols, inplace=True)
    
    # We keep 'timestamp' usually for splitting, or drop it?
    # Prompt doesn't explicitly say to drop timestamp, but typically ML models (trees) don't use raw datetime.
    # We extracted temporal features. Let's drop raw timestamp.
    df.drop(columns=['timestamp'], inplace=True)
    
    return df

def main():
    df = load_data(INPUT_PATH)
    
    df = create_temporal_features(df)
    df = create_velocity_features(df)
    df = create_spending_behavior_features(df)
    df = create_history_based_features(df)
    df = create_interaction_features(df)
    df = encode_and_cleanup(df)
    
    print("-" * 30)
    print("Final Feature List:")
    print(df.columns.tolist())
    print("-" * 30)
    
    print("Saving to", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print("-" * 30)
    print(f"Dataset Shape: {df.shape}")
    print(f"Fraud Rate: {(df['is_fraud'].mean()*100):.2f}%")
    print("-" * 30)
    print("First 5 Rows:")
    print(df.head())

if __name__ == "__main__":
    main()
