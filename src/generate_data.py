import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# --- Configuration ---
TOTAL_TRANSACTIONS_MIN = 15000
FRAUD_RATE_TARGET = 0.03  # Aim for ~3% fraud
NUM_USERS = 300
NUM_MERCHANTS = 60
SIMULATION_DAYS = 45 # Increased days to spread out transactions
START_DATE = datetime.now() - timedelta(days=SIMULATION_DAYS)

# --- Constants ---
COUNTRIES = ['US', 'UK', 'IN', 'CA', 'DE', 'FR', 'AU', 'JP']
PAYMENT_METHODS = ['Credit Card', 'Debit Card', 'UPI', 'Digital Wallet', 'Net Banking']
CURRENCIES = {'US': 'USD', 'UK': 'GBP', 'IN': 'INR', 'CA': 'CAD', 'DE': 'EUR', 'FR': 'EUR', 'AU': 'AUD', 'JP': 'JPY'}

CATEGORIES = [
    'es_transportation', 'es_food', 'es_health', 'es_wellnessandbeauty', 
    'es_fashion', 'es_barsandrestaurants', 'es_hyper', 'es_sportsandtoys', 
    'es_tech', 'es_home', 'es_hotelservices', 'es_otherservices', 
    'es_contents', 'es_travel', 'es_leisure'
]

# Categories with specific behaviors
LOW_VALUE_CATEGORIES = ['es_transportation', 'es_food']
HIGH_VALUE_CATEGORIES = ['es_travel', 'es_leisure']

# --- Entity Generation ---

class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.home_country = random.choice(COUNTRIES)
        self.preferred_payment_methods = random.sample(PAYMENT_METHODS, k=random.randint(1, 2))
        
        # Demographics
        self.age = random.randint(18, 75)
        self.gender = random.choice(['M', 'F'])
        
        # Spending profile (Baseline)
        self.avg_spend_baseline = np.random.lognormal(mean=3.5, sigma=1.0)
        self.frequent_merchants = [] 
        self.devices = [f"device_{uuid.uuid4().hex[:8]}" for _ in range(random.randint(1, 2))]

    def get_transaction_amount(self, category):
        # Base amount from user profile
        base = self.avg_spend_baseline
        
        if category in LOW_VALUE_CATEGORIES:
            # Low value: $5 - $20 range typically
            mu = 10
            sigma = 5
            amount = max(1.0, np.random.normal(mu, sigma))
        elif category in HIGH_VALUE_CATEGORIES:
            # High value: $100 - $1000+
            mu = base * 5 
            sigma = base * 2
            amount = max(50.0, np.random.normal(mu, sigma))
        else:
            # Medium value: Cluster around base
            amount = max(5.0, np.random.normal(base, base * 0.3))
            
        return round(amount, 2)

# Generate Merchant Pool with Categories
merchants = []
for i in range(NUM_MERCHANTS):
    m_id = f"M_{i:04d}"
    # Assign category - weight slightly away from travel/leisure so they are rarer
    cat = random.choices(CATEGORIES, weights=[
        15, 15, 5, 5, 
        10, 10, 10, 5, 
        5, 5, 5, 5, 
        2, 2, 2
    ], k=1)[0]
    merchants.append({'id': m_id, 'category': cat})

# Helper to get merchant object
def get_merchant(m_id):
    for m in merchants:
        if m['id'] == m_id:
            return m
    return None

users = [User(f"U_{i:04d}") for i in range(NUM_USERS)]

# Assign frequent merchants to users (mostly local/daily stuff)
for user in users:
    # Users prefer shops in "food", "transport", "hyper", "wellness"
    daily_merchants = [m for m in merchants if m['category'] in ['es_food', 'es_transportation', 'es_hyper', 'es_barsandrestaurants']]
    if daily_merchants:
        user.frequent_merchants = random.sample(daily_merchants, k=min(len(daily_merchants), random.randint(3, 8)))
    else:
        user.frequent_merchants = random.sample(merchants, k=3)

# --- Data Containers ---
transactions = []

# --- Helper Functions ---

def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def add_transaction(user, timestamp, amount, merchant_info=None, device=None, 
                    location=None, payment_method=None, is_fraud=0, pattern_desc="Normal"):
    
    if merchant_info is None:
        # 80% chance of frequent merchant, 20% random
        if random.random() < 0.8 and user.frequent_merchants:
            merchant_info = random.choice(user.frequent_merchants)
        else:
            merchant_info = random.choice(merchants)
            
    if device is None:
        device = random.choice(user.devices)
        
    if location is None:
        location = user.home_country
        
    if payment_method is None:
        payment_method = random.choice(user.preferred_payment_methods)

    # Recalculate amount if it wasn't strictly enforced, to match category
    # But usually the caller calculates amount. If amount is passed as 0 or None (not happens here), we'd fix it.
    # We assume 'amount' passed roughly aligns, but if we just picked a random merchant_info, 
    # we should probably re-verify amount fits category IF it was auto-generated.
    # For now, we trust the flow for Normal txs.
    
    transactions.append({
        'transaction_id': '', # To be filled later
        'user_id': user.user_id,
        'merchant_id': merchant_info['id'],
        'category': merchant_info['category'],
        'amount': amount,
        'timestamp': timestamp,
        'location': location,
        'payment_method': payment_method,
        'device_id': device,
        'age': user.age,
        'gender': user.gender,
        'is_fraud': is_fraud,
        'pattern_desc': pattern_desc
    })

# --- 1. Generate Normal Transactions ---
print("Generating normal transactions...")

for user in users:
    # Generate around 50-80 transactions per user
    num_tx = int(np.random.normal(70, 20))
    if num_tx < 10: num_tx = 10
    
    current_time = START_DATE
    
    for _ in range(num_tx):
        # Time gaps
        time_gap_minutes = np.random.exponential(scale=1000) # ~16 hours avg
        current_time += timedelta(minutes=time_gap_minutes)
        
        if current_time > datetime.now():
            break
        
        # Select merchant first to determine amount
        if random.random() < 0.8 and user.frequent_merchants:
            merchant = random.choice(user.frequent_merchants)
        else:
            merchant = random.choice(merchants)
            
        amt = user.get_transaction_amount(merchant['category'])
        
        add_transaction(
            user=user,
            timestamp=current_time,
            amount=amt,
            merchant_info=merchant,
            is_fraud=0,
            pattern_desc="Normal"
        )

# --- 2. Inject Fraud Patterns ---
print("Injecting fraud patterns...")

# We have approx 15k normal tx. Need ~3% fraud => ~450 fraud tx.
FRAUD_COUNT = 0
TARGET_FRAUD = 500

# Helper to find realistic target merchants for fraud (excluding low value ones)
fraud_susceptible_merchants = [m for m in merchants if m['category'] not in LOW_VALUE_CATEGORIES]
high_value_merchants = [m for m in merchants if m['category'] in HIGH_VALUE_CATEGORIES]

# --- Pattern 1: Unusual Transaction Velocity ---
# Burst of transactions
for _ in range(30): 
    target_user = random.choice(users)
    start_time = random_date(START_DATE, datetime.now())
    
    # 5 to 12 rapid transactions
    for i in range(random.randint(5, 12)):
        tx_time = start_time + timedelta(seconds=i*45)
        # Use random susceptible merchant
        target_merchant = random.choice(fraud_susceptible_merchants)
        amt = target_user.get_transaction_amount(target_merchant['category'])
        
        add_transaction(
            user=target_user,
            timestamp=tx_time,
            amount=amt,
            merchant_info=target_merchant,
            is_fraud=1,
            pattern_desc="Pattern 1: Velocity"
        )

# --- Pattern 2: Sudden Amount Spikes ---
# High value compared to history
for _ in range(60):
    target_user = random.choice(users)
    tx_time = random_date(START_DATE, datetime.now())
    
    # Pick a high value category merchant (e.g., tech, jewelry, travel)
    # If we don't have enough, pick any susceptible
    target_m = random.choice(high_value_merchants) if high_value_merchants else random.choice(fraud_susceptible_merchants)
    
    # Spike amount: 5x to 10x of user's average baseline
    spike_amount = target_user.avg_spend_baseline * random.uniform(5, 10)
    # Ensure it's substantial
    if spike_amount < 200: spike_amount = random.uniform(200, 1000)
    
    add_transaction(
        user=target_user,
        timestamp=tx_time,
        amount=spike_amount,
        merchant_info=target_m,
        is_fraud=1,
        pattern_desc="Pattern 2: Amount Spike"
    )

# --- Pattern 3: Location Inconsistency ---
# Transact in foreign country shortly after normal activity
for _ in range(40):
    target_user = random.choice(users)
    base_time = random_date(START_DATE, datetime.now())
    
    # Normal Tx
    m_normal = random.choice(merchants)
    add_transaction(
        user=target_user,
        timestamp=base_time,
        amount=target_user.get_transaction_amount(m_normal['category']),
        merchant_info=m_normal,
        is_fraud=0,
        pattern_desc="Normal (Pre-Pattern 3)"
    )
    
    # Fraud Tx
    fraud_time = base_time + timedelta(minutes=random.randint(30, 120))
    foreign_loc = random.choice([c for c in COUNTRIES if c != target_user.home_country])
    m_fraud = random.choice(fraud_susceptible_merchants)
    
    add_transaction(
        user=target_user,
        timestamp=fraud_time,
        amount=target_user.get_transaction_amount(m_fraud['category']) * 1.5,
        merchant_info=m_fraud,
        location=foreign_loc,
        is_fraud=1,
        pattern_desc="Pattern 3: Location Inconsistency"
    )

# --- Pattern 4: Repeated Merchant Abuse ---
# One merchant gets hit hard
victim_merchant = random.choice(fraud_susceptible_merchants)
for _ in range(15): # 15 separate attack sequences on this merchant
    target_user = random.choice(users)
    start_time = random_date(START_DATE, datetime.now())
    
    for i in range(random.randint(3, 6)):
        tx_time = start_time + timedelta(minutes=i*5)
        amt = random.uniform(100, 300) # Fixed abuse amount range
        add_transaction(
            user=target_user,
            timestamp=tx_time,
            amount=amt,
            merchant_info=victim_merchant,
            is_fraud=1,
            pattern_desc="Pattern 4: Merchant Abuse"
        )

# --- Pattern 5: Shared Device Across Users ---
bad_device = "device_EMULATOR_X99"
start_time = random_date(START_DATE, datetime.now())
for i in range(25): # 25 different users
    target_user = random.choice(users)
    tx_time = start_time + timedelta(minutes=random.randint(0, 1000))
    m_fraud = random.choice(fraud_susceptible_merchants)
    
    add_transaction(
        user=target_user,
        timestamp=tx_time,
        amount=target_user.get_transaction_amount(m_fraud['category']),
        merchant_info=m_fraud,
        device=bad_device,
        is_fraud=1,
        pattern_desc="Pattern 5: Shared Device"
    )

# --- Pattern 6: Payment Method Switching ---
for _ in range(30):
    target_user = random.choice(users)
    start_time = random_date(START_DATE, datetime.now())
    methods = random.sample(PAYMENT_METHODS, 3)
    
    for i, meth in enumerate(methods):
        tx_time = start_time + timedelta(minutes=i*3)
        m_fraud = random.choice(fraud_susceptible_merchants)
        add_transaction(
            user=target_user,
            timestamp=tx_time,
            amount=random.uniform(20, 100),
            merchant_info=m_fraud,
            payment_method=meth,
            is_fraud=1,
            pattern_desc="Pattern 6: Payment Switching"
        )

# --- Pattern 7: Dormant Account Reactivation ---
# Pick users with low activity count in 'transactions' list so far
# (Approximation roughly)
curr_counts = {}
for t in transactions:
    curr_counts[t['user_id']] = curr_counts.get(t['user_id'], 0) + 1

dormant_candidates = [u for u in users if curr_counts.get(u.user_id, 0) < 15]
if not dormant_candidates:
    dormant_candidates = random.sample(users, 10)

for user in dormant_candidates[:10]:
    # Hit them at the very end of simulation
    burst_time = datetime.now() - timedelta(hours=5)
    for i in range(6):
        tx_time = burst_time + timedelta(minutes=i*10)
        m_fraud = random.choice(fraud_susceptible_merchants)
        add_transaction(
            user=user,
            timestamp=tx_time,
            amount=random.uniform(200, 500),
            merchant_info=m_fraud,
            is_fraud=1,
            pattern_desc="Pattern 7: Dormant Reactivation"
        )

# --- Pattern 8: Small-Then-Large Transaction Escalation ---
for _ in range(25):
    target_user = random.choice(users)
    base_time = random_date(START_DATE, datetime.now())
    
    # 1. Very small (potentially safe category)
    m_safe = random.choice([m for m in merchants if m['category'] in LOW_VALUE_CATEGORIES])
    add_transaction(
        user=target_user,
        timestamp=base_time,
        amount=random.uniform(1, 5),
        merchant_info=m_safe,
        is_fraud=1,
        pattern_desc="Pattern 8: Small Test"
    )
    
    # 2. Large drain (unsafe category)
    m_drain = random.choice(fraud_susceptible_merchants)
    drain_time = base_time + timedelta(minutes=15)
    add_transaction(
        user=target_user,
        timestamp=drain_time,
        amount=random.uniform(500, 2000),
        merchant_info=m_drain,
        is_fraud=1,
        pattern_desc="Pattern 8: Large Drain"
    )


# --- Finalize and Save ---

df = pd.DataFrame(transactions)
df.sort_values(by='timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)
df['transaction_id'] = [f"TX_{i:07d}" for i in range(len(df))]

# Reorder and Select Columns
cols = [
    'transaction_id', 'user_id', 'merchant_id', 'category', 'amount', 
    'timestamp', 'location', 'payment_method', 'device_id', 
    'age', 'gender', 'is_fraud'
]
final_df = df[cols]

# Statistics
total_tx = len(final_df)
fraud_tx = final_df['is_fraud'].sum()
fraud_rate = (fraud_tx / total_tx) * 100

print("=" * 40)
print("DATASET GENERATION REPORT")
print("=" * 40)
print(f"Total Transactions: {total_tx}")
print(f"Fraud Transactions: {fraud_tx}")
print(f"Fraud Percentage:   {fraud_rate:.2f}%")
print("-" * 40)
print("Fraud Distribution by Category:")
print(final_df[final_df['is_fraud'] == 1]['category'].value_counts().head(10))
print("-" * 40)

# Save
output_path = "data/transactions.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")
