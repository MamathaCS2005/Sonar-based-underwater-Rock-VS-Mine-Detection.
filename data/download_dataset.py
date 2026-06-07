"""
Download the Sonar Rock vs Mine dataset from UCI repository
and save it as data/sonar.csv
"""
import pandas as pd
import os

def download_sonar_dataset():
    """Download sonar dataset from UCI ML Repository."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    
    # Column names: 60 features + 1 label
    columns = [f"feature_{i+1}" for i in range(60)] + ["label"]
    
    print("Downloading Sonar dataset from UCI repository...")
    df = pd.read_csv(url, header=None, names=columns)
    
    # Save to data directory
    output_path = os.path.join(os.path.dirname(__file__), "sonar.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Classes: {df['label'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    download_sonar_dataset()
