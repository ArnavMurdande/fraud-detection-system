import pandas as pd
from neo4j import GraphDatabase, basic_auth
import os
import sys

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password") # Default, change if needed
DATA_PATH = "data/transactions.csv"
OUTPUT_FILE = "results/graph_analysis.txt"

class FraudGraph:
    def __init__(self, uri, user, password):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            self.verify_connection()
        except Exception as e:
            print(f"Failed to connect to Neo4j at {uri}: {e}")
            print("Please ensure Neo4j is running and credentials are correct.")
            sys.exit(1)

    def close(self):
        if self.driver:
            self.driver.close()

    def verify_connection(self):
        with self.driver.session() as session:
            session.run("RETURN 1")
            print("Connected to Neo4j successfully.")

    def clear_database(self):
        print("Clearing existing graph data...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def ingest_data(self, csv_path, limit=None):
        print(f"Ingesting data from {csv_path}...")
        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)
        
        # Ingest in batches or standard loop? 
        # For simplicity in this script, we'll iterate. For prod, use LOAD CSV or batched params.
        
        query = """
        MERGE (u:User {id: $user_id})
        MERGE (d:Device {id: $device_id})
        MERGE (m:Merchant {id: $merchant_id})
        MERGE (l:Location {name: $location})
        
        MERGE (u)-[:USES]->(d)
        MERGE (u)-[:PAYS_AT]->(m)
        MERGE (u)-[:LOCATED_IN]->(l)
        
        CREATE (tx:Transaction {
            id: $tx_id,
            amount: $amount,
            timestamp: $timestamp,
            is_fraud: $is_fraud
        })
        CREATE (u)-[:PERFORMED]->(tx)
        CREATE (tx)-[:TO]->(m)
        """
        
        with self.driver.session() as session:
            count = 0
            for index, row in df.iterrows():
                # Provide defaults if cols missing
                params = {
                    "user_id": str(row.get('user_id', 'unknown')),
                    "device_id": str(row.get('device_id', 'unknown')),
                    "merchant_id": str(row.get('merchant_id', 'unknown')),
                    "location": str(row.get('location', 'unknown')),
                    "tx_id": str(index), # Using index as tx_id if not present
                    "amount": float(row.get('amount', 0)),
                    "timestamp": str(row.get('timestamp', '')),
                    "is_fraud": int(row.get('is_fraud', 0))
                }
                session.run(query, params)
                count += 1
                if count % 1000 == 0:
                    print(f"Ingested {count} transactions...")
        print(f"Ingestion complete. Total: {len(df)}")

    def run_analysis(self):
        results = []
        
        # 1. Device Shared Count (Users per Device)
        print("Analyzing Shared Devices...")
        q1 = """
        MATCH (u:User)-[:USES]->(d:Device)
        WITH d, count(distinct u) as user_count
        WHERE user_count > 1
        RETURN d.id as device, user_count
        ORDER BY user_count DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            records = session.run(q1)
            results.append("TOP SHARED DEVICES (Potential Fraud Rings):")
            for r in records:
                results.append(f"Device: {r['device']} | Users: {r['user_count']}")
            results.append("-" * 30)

    
        # 2. Merchant Fraud Density
        print("Analyzing Merchant Fraud Density...")
        q2 = """
        MATCH (tx:Transaction)-[:TO]->(m:Merchant)
        WITH m, count(tx) as total_tx, sum(tx.is_fraud) as fraud_tx
        WHERE total_tx > 5 AND fraud_tx > 0
        RETURN m.id as merchant, total_tx, fraud_tx, (toFloat(fraud_tx)/total_tx) as fraud_rate
        ORDER BY fraud_rate DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            records = session.run(q2)
            results.append("HIGH RISK MERCHANTS:")
            for r in records:
                results.append(f"Merchant: {r['merchant']} | Total: {r['total_tx']} | Fraud: {r['fraud_tx']} | Rate: {r['fraud_rate']:.2f}")
            results.append("-" * 30)
            
        # 3. Graph Risk Score (Simple Degree Centrality of Fraud neighbors)
        # Find users connected to confirmed fraud cases via Device
        print("Calculating Graph Risk Features...")
        q3 = """
        MATCH (u:User)-[:USES]->(d:Device)<-[:USES]-(other:User)-[:PERFORMED]->(tx:Transaction)
        WHERE tx.is_fraud = 1 AND u <> other
        RETURN u.id as user, count(distinct tx) as connected_frauds
        ORDER BY connected_frauds DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            records = session.run(q3)
            results.append("USERS CONNECTED TO KNOWN FRAUD (via Shared Device):")
            for r in records:
                results.append(f"User: {r['user']} | Linked Fraud Cases: {r['connected_frauds']}")
            results.append("-" * 30)

        return results

    def save_results(self, lines):
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            f.write("NEO4J GRAPH ANALYSIS REPORT\n")
            f.write("===========================\n")
            f.write("\n".join(lines))
            
            f.write("\n\nMODEL INTEGRATION NOTES:\n")
            f.write("Graph-based features (e.g., 'connected_frauds', 'device_user_degree') \n")
            f.write("provide relational context that tabular models often miss.\n")
            f.write("Strategy:\n")
            f.write("1. Compute graph features (e.g. PageRank, Community ID) in Neo4j.\n")
            f.write("2. Export these features to the CSV.\n")
            f.write("3. Retrain XGBoost with these new 'Risk' columns.\n")
            f.write("This creates a feedback loop: ML detects pattern -> Graph connects entities -> Stronger ML features.\n")
            
        print(f"Analysis saved to {OUTPUT_FILE}")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data file {DATA_PATH} not found.")
        sys.exit(1)
        
    # Attempt connection
    # Note: This requires a running Neo4j instance.
    graph = FraudGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        graph.clear_database()
        # Limit ingestion for demo speed if needed, or full
        graph.ingest_data(DATA_PATH, limit=5000) 
        results = graph.run_analysis()
        graph.save_results(results)
        
        print("\nTop 3 Analysis Lines:")
        for line in results[:3]:
            print(line)
            
    finally:
        graph.close()

if __name__ == "__main__":
    main()
