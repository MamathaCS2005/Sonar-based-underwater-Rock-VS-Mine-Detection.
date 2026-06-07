"""
Create the Sonar dataset locally.
Uses the well-known Sonar Mines vs Rocks dataset values.
"""
import os

# The sonar dataset - 208 samples, 60 features + label
# Source: UCI ML Repository - Connectionist Bench (Sonar, Mines vs. Rocks)
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"

def create_dataset():
    """Download from GitHub mirror (more reliable than UCI)."""
    import pandas as pd
    
    print("Downloading Sonar dataset...")
    columns = [f"feature_{i+1}" for i in range(60)] + ["label"]
    
    try:
        df = pd.read_csv(DATA_URL, header=None, names=columns)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Generating synthetic sonar-like dataset instead...")
        df = generate_synthetic_dataset()
    
    output_path = os.path.join(os.path.dirname(__file__), "sonar.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Classes: {df['label'].value_counts().to_dict()}")
    return df


def generate_synthetic_dataset():
    """Generate a synthetic dataset matching sonar characteristics if download fails."""
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n_mines = 111
    n_rocks = 97
    n_features = 60
    
    # Mines tend to have higher energy in mid-frequency bands
    mine_data = np.random.uniform(0.0, 0.6, (n_mines, n_features))
    mine_data[:, 10:40] += np.random.uniform(0.1, 0.4, (n_mines, 30))
    mine_data = np.clip(mine_data, 0, 1)
    
    # Rocks have more uniform, generally lower energy
    rock_data = np.random.uniform(0.0, 0.5, (n_rocks, n_features))
    rock_data[:, 5:20] += np.random.uniform(0.05, 0.2, (n_rocks, 15))
    rock_data = np.clip(rock_data, 0, 1)
    
    columns = [f"feature_{i+1}" for i in range(n_features)] + ["label"]
    
    mine_df = pd.DataFrame(mine_data, columns=columns[:-1])
    mine_df["label"] = "M"
    
    rock_df = pd.DataFrame(rock_data, columns=columns[:-1])
    rock_df["label"] = "R"
    
    df = pd.concat([mine_df, rock_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    create_dataset()
